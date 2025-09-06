from typing import Dict, List, Tuple, Any
import heapq


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

    # Ensure task stations and s0 exist in graph, even if isolated in input
    nodes.add(s0)
    for t in tasks:
        nodes.add(t["station"])  # type: ignore[index]

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
    # Any target not found remains INF
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

    # Build graph and compute distances between key nodes (task stations + s0)
    adj = _build_graph(subway, tasks, s0)
    key_nodes = sorted({t["station"] for t in tasks} | {s0})
    dists = _all_pairs_shortest_on_keys(adj, key_nodes)

    # Helper to get distance between two key stations
    def d(u: int, v: int) -> int:
        if u == v:
            return 0
        return dists[u][v]

    # Sort tasks by end time, then by start time to stabilize
    tasks_sorted = sorted(tasks, key=lambda x: (x["end"], x["start"]))

    n = len(tasks_sorted)
    # Prepare arrays for DP over schedules ending at i
    best_score = [0] * n
    best_fee = [0] * n
    prev_idx = [-1] * n

    # Precompute station and name arrays for speed
    station = [t["station"] for t in tasks_sorted]
    start = [t["start"] for t in tasks_sorted]
    end = [t["end"] for t in tasks_sorted]
    score = [t["score"] for t in tasks_sorted]
    name = [t["name"] for t in tasks_sorted]

    # Base and transitions
    for i in range(n):
        si = station[i]
        # Base: only task i
        cur_score = score[i]
        cur_fee = d(s0, si) + d(si, s0)
        cur_prev = -1

        # Consider any predecessor j with end_j <= start_i
        for j in range(i):
            if end[j] <= start[i]:
                sj = station[j]
                cand_score = best_score[j] + score[i]
                # Update fee by removing return from sj->s0, adding sj->si + si->s0
                cand_fee = best_fee[j] - d(sj, s0) + d(sj, si) + d(si, s0)

                if cand_score > cur_score or (cand_score == cur_score and cand_fee < cur_fee):
                    cur_score = cand_score
                    cur_fee = cand_fee
                    cur_prev = j

        best_score[i] = cur_score
        best_fee[i] = cur_fee
        prev_idx[i] = cur_prev

    # Choose best among all endings and the empty schedule
    global_best_score = 0
    global_best_fee = 0
    last_idx = -1

    for i in range(n):
        if best_score[i] > global_best_score or (best_score[i] == global_best_score and best_fee[i] < global_best_fee) or last_idx == -1 and best_score[i] > 0:
            global_best_score = best_score[i]
            global_best_fee = best_fee[i]
            last_idx = i

    if global_best_score == 0:
        return {"max_score": 0, "min_fee": 0, "schedule": []}

    # Reconstruct schedule
    selected_indices: List[int] = []
    i = last_idx
    while i != -1:
        selected_indices.append(i)
        i = prev_idx[i]
    selected_indices.reverse()

    # Sort final schedule by start time ascending (as required)
    selected_indices.sort(key=lambda idx: start[idx])
    schedule_names = [name[idx] for idx in selected_indices]

    return {
        "max_score": int(global_best_score),
        "min_fee": int(global_best_fee),
        "schedule": schedule_names,
    }