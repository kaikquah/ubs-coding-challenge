from typing import Dict, List, Tuple, Any
import heapq
import bisect
from collections import defaultdict


INF = 10 ** 18


def _build_graph(subway: List[Dict[str, Any]], tasks: List[Dict[str, Any]], s0: int) -> Dict[int, List[Tuple[int, int]]]:
    nodes = set()
    adj: Dict[int, List[Tuple[int, int]]] = {}

    # Collect nodes from subway
    for route in subway:
        u, v = route["connection"]
        w = route["fee"]
        nodes.add(u)
        nodes.add(v)
        adj.setdefault(u, []).append((v, w))
        adj.setdefault(v, []).append((u, w))

    # Ensure task stations and s0 exist in graph
    nodes.add(s0)
    for t in tasks:
        nodes.add(t["station"])

    for node in nodes:
        adj.setdefault(node, [])

    return adj


def _dijkstra_from(src: int, adj: Dict[int, List[Tuple[int, int]]], targets: set) -> Dict[int, int]:
    dist: Dict[int, int] = {src: 0}
    h: List[Tuple[int, int]] = [(0, src)]
    found: Dict[int, int] = {}
    remaining_targets = set(targets)

    while h and remaining_targets:
        d, u = heapq.heappop(h)
        if d != dist.get(u, INF):
            continue
        if u in remaining_targets:
            found[u] = d
            remaining_targets.remove(u)
            if not remaining_targets:
                break
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                heapq.heappush(h, (nd, v))
    
    # Set unfound targets to INF
    for t in targets:
        found.setdefault(t, INF)
    return found


def _all_pairs_shortest_on_keys(adj: Dict[int, List[Tuple[int, int]]], key_nodes: List[int]) -> Dict[int, Dict[int, int]]:
    key_set = set(key_nodes)
    dists: Dict[int, Dict[int, int]] = {}
    for u in key_nodes:
        dists[u] = _dijkstra_from(u, adj, key_set)
    return dists


def solve_princess_diaries(payload: Dict[str, Any]) -> Dict[str, Any]:
    tasks_in: List[Dict[str, Any]] = payload.get("tasks", [])
    subway: List[Dict[str, Any]] = payload.get("subway", [])
    s0 = payload.get("starting_station")
    if s0 is None:
        raise ValueError("starting_station is required")
    s0 = int(s0)

    # Normalize and validate tasks
    tasks: List[Dict[str, Any]] = []
    for t in tasks_in:
        try:
            tasks.append({
                "name": t["name"],
                "start": int(t["start"]),
                "end": int(t["end"]),
                "station": int(t["station"]),
                "score": int(t["score"]),
            })
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid task data: {e}")

    # Edge case: no tasks
    if not tasks:
        return {"max_score": 0, "min_fee": 0, "schedule": []}

    # Build graph and compute distances between key nodes
    adj = _build_graph(subway, tasks, s0)
    key_nodes = sorted({t["station"] for t in tasks} | {s0})
    dists = _all_pairs_shortest_on_keys(adj, key_nodes)

    # Helper to get distance between stations
    def d(u: int, v: int) -> int:
        if u == v:
            return 0
        return dists[u][v]

    # OPTIMIZATION 1: Sort tasks by end time for better DP efficiency
    indexed_tasks = [(i, t) for i, t in enumerate(tasks)]
    indexed_tasks.sort(key=lambda x: (x[1]["end"], x[1]["start"]))
    
    n = len(tasks)
    
    # OPTIMIZATION 2: Use sweep line algorithm with efficient data structures
    # Process tasks chronologically and maintain efficient predecessor lookup
    
    # For each task index in original array, store DP result
    dp_results = {}  # orig_index -> (best_score, best_fee, prev_orig_index)
    
    # Maintain list of processed tasks for efficient predecessor queries
    # Format: (end_time, best_score, best_fee, orig_index)
    processed_tasks = []
    
    for sorted_idx, (orig_idx, task) in enumerate(indexed_tasks):
        station_i = task["station"]
        start_i = task["start"]
        end_i = task["end"]
        score_i = task["score"]
        name_i = task["name"]
        
        # Base case: only current task
        best_score = score_i
        best_fee = d(s0, station_i) + d(station_i, s0)
        best_prev = -1
        
        # OPTIMIZATION 3: Binary search to find valid predecessors
        # Find all processed tasks that ended before current task starts
        search_key = (start_i + 1, -1, -1, -1)  # +1 to find tasks with end <= start_i
        valid_count = bisect.bisect_left(processed_tasks, search_key)
        
        # OPTIMIZATION 4: Only check valid predecessors (much fewer in most cases)
        for j in range(valid_count):
            end_j, score_j, fee_j, orig_j = processed_tasks[j]
            if end_j <= start_i:  # Double check due to binary search approximation
                prev_task = tasks[orig_j]
                station_j = prev_task["station"]
                
                # Calculate transition cost
                cand_score = score_j + score_i
                # Remove return to s0 from prev, add transition cost, add new return to s0
                cand_fee = fee_j - d(station_j, s0) + d(station_j, station_i) + d(station_i, s0)
                
                # Update if better
                if (cand_score > best_score or 
                    (cand_score == best_score and cand_fee < best_fee)):
                    best_score = cand_score
                    best_fee = cand_fee
                    best_prev = orig_j
        
        # Store result for this task
        dp_results[orig_idx] = (best_score, best_fee, best_prev)
        
        # OPTIMIZATION 5: Insert in sorted order for next binary searches
        new_entry = (end_i, best_score, best_fee, orig_idx)
        bisect.insort(processed_tasks, new_entry)
    
    # Find global optimum among all possible ending tasks
    global_best_score = 0
    global_best_fee = 0
    best_ending_task = -1
    
    for orig_idx in range(n):
        score_val, fee_val, _ = dp_results[orig_idx]
        if (score_val > global_best_score or 
            (score_val == global_best_score and fee_val < global_best_fee) or
            (best_ending_task == -1 and score_val > 0)):
            global_best_score = score_val
            global_best_fee = fee_val
            best_ending_task = orig_idx
    
    if global_best_score == 0:
        return {"max_score": 0, "min_fee": 0, "schedule": []}
    
    # Reconstruct optimal schedule
    schedule_indices = []
    current = best_ending_task
    while current != -1:
        schedule_indices.append(current)
        _, _, prev = dp_results[current]
        current = prev
    
    schedule_indices.reverse()
    
    # Sort final schedule by start time (as required by problem)
    schedule_indices.sort(key=lambda idx: tasks[idx]["start"])
    schedule_names = [tasks[idx]["name"] for idx in schedule_indices]
    
    return {
        "max_score": int(global_best_score),
        "min_fee": int(global_best_fee),
        "schedule": schedule_names,
    }