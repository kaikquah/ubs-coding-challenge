import heapq
from typing import Dict, List, Tuple, Any

def dijkstra_single_source(adj: List[List[Tuple[int, int]]], source: int, n: int) -> List[int]:
    """Optimized Dijkstra using array-based adjacency list"""
    dist = [float('inf')] * n
    dist[source] = 0
    heap = [(0, source)]
    
    while heap:
        d, u = heapq.heappop(heap)
        
        if d > dist[u]:
            continue
            
        for v, weight in adj[u]:
            new_dist = d + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
    
    return dist

def solve_princess_diaries(payload: Dict[str, Any]) -> Dict[str, Any]:
    tasks = payload.get("tasks", [])
    subway = payload.get("subway", [])
    s0 = payload.get("starting_station")
    
    if not tasks:
        return {"max_score": 0, "min_fee": 0, "schedule": []}
    
    n = len(tasks)
    
    # Map station IDs to compact indices for efficiency
    station_set = {s0}
    for task in tasks:
        station_set.add(task["station"])
    
    station_to_idx = {station: idx for idx, station in enumerate(sorted(station_set))}
    idx_to_station = {idx: station for station, idx in station_to_idx.items()}
    num_stations = len(station_set)
    
    # Build adjacency list with compact indices
    adj = [[] for _ in range(num_stations)]
    for route in subway:
        u, v = route["connection"]
        fee = route["fee"]
        if u in station_to_idx and v in station_to_idx:
            u_idx = station_to_idx[u]
            v_idx = station_to_idx[v]
            adj[u_idx].append((v_idx, fee))
            adj[v_idx].append((u_idx, fee))
    
    # Precompute distances from s0 and all task stations
    s0_idx = station_to_idx[s0]
    dist_from_s0 = dijkstra_single_source(adj, s0_idx, num_stations)
    
    # Store task station indices
    task_stations_idx = []
    for task in tasks:
        task_stations_idx.append(station_to_idx[task["station"]])
    
    # Compute distances from each unique task station
    unique_task_stations = set(task_stations_idx)
    dist_from_stations = {}
    for station_idx in unique_task_stations:
        dist_from_stations[station_idx] = dijkstra_single_source(adj, station_idx, num_stations)
    
    # Sort tasks by end time
    indexed_tasks = [(i, task) for i, task in enumerate(tasks)]
    indexed_tasks.sort(key=lambda x: (x[1]["end"], x[1]["start"]))
    
    # DP with optimized distance lookups
    dp = [(0, 0, -1)] * n  # (score, fee, parent_idx)
    
    for i in range(n):
        orig_idx_i, task_i = indexed_tasks[i]
        station_idx_i = task_stations_idx[orig_idx_i]
        score_i = task_i["score"]
        start_i = task_i["start"]
        
        # Base case: direct route from s0 to task and back
        fee_to = dist_from_s0[station_idx_i]
        fee_back = dist_from_s0[station_idx_i]  # Same distance back
        base_fee = fee_to + fee_back
        
        dp[i] = (score_i, base_fee, -1)
        
        # Try extending from compatible predecessors
        for j in range(i):
            orig_idx_j, task_j = indexed_tasks[j]
            
            if task_j["end"] <= start_i:
                station_idx_j = task_stations_idx[orig_idx_j]
                prev_score, prev_fee, _ = dp[j]
                
                # Calculate transition cost
                # Remove: station_j -> s0 from prev_fee
                # Add: station_j -> station_i -> s0
                fee_j_to_s0 = dist_from_stations[station_idx_j][s0_idx]
                fee_j_to_i = dist_from_stations[station_idx_j][station_idx_i]
                fee_i_to_s0 = dist_from_s0[station_idx_i]
                
                new_score = prev_score + score_i
                new_fee = prev_fee - fee_j_to_s0 + fee_j_to_i + fee_i_to_s0
                
                # Update if better
                if new_score > dp[i][0] or (new_score == dp[i][0] and new_fee < dp[i][1]):
                    dp[i] = (new_score, new_fee, j)
    
    # Find best solution
    best_score = 0
    best_fee = float('inf')
    best_idx = -1
    
    for i in range(n):
        score, fee, _ = dp[i]
        if score > best_score or (score == best_score and fee < best_fee):
            best_score = score
            best_fee = fee
            best_idx = i
    
    if best_score == 0:
        return {"max_score": 0, "min_fee": 0, "schedule": []}
    
    # Reconstruct schedule
    schedule = []
    current = best_idx
    while current != -1:
        orig_idx, task = indexed_tasks[current]
        schedule.append(task["name"])
        current = dp[current][2]
    
    schedule.reverse()
    
    # Sort by start time for output
    task_map = {t["name"]: t for t in tasks}
    schedule.sort(key=lambda name: task_map[name]["start"])
    
    return {
        "max_score": int(best_score),
        "min_fee": int(best_fee),
        "schedule": schedule
    }