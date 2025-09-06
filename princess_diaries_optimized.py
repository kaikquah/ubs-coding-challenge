import heapq
import bisect
from collections import defaultdict
from typing import Dict, List, Tuple, Any

INF = float('inf')

def dijkstra(graph: Dict[int, List[Tuple[int, int]]], start: int, targets: set) -> Dict[int, int]:
    """Optimized Dijkstra that stops when all targets are found"""
    dist = {start: 0}
    heap = [(0, start)]
    found = {}
    remaining = set(targets)
    
    while heap and remaining:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, INF):
            continue
            
        if u in remaining:
            found[u] = d
            remaining.remove(u)
            if not remaining:
                break
                
        for v, weight in graph.get(u, []):
            new_dist = d + weight
            if new_dist < dist.get(v, INF):
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
    
    for target in targets:
        found.setdefault(target, INF)
    
    return found

def build_graph(subway: List[Dict[str, Any]], key_stations: set) -> Dict[int, List[Tuple[int, int]]]:
    """Build adjacency list for the subway graph"""
    graph = defaultdict(list)
    
    for route in subway:
        u, v = route["connection"]
        fee = route["fee"]
        graph[u].append((v, fee))
        graph[v].append((u, fee))
    
    for station in key_stations:
        if station not in graph:
            graph[station] = []
    
    return dict(graph)

def solve_princess_diaries(payload: Dict[str, Any]) -> Dict[str, Any]:
    tasks = payload.get("tasks", [])
    subway = payload.get("subway", [])
    s0 = payload.get("starting_station")
    
    if not tasks:
        return {"max_score": 0, "min_fee": 0, "schedule": []}
    
    # Sort tasks by end time for optimal DP ordering
    tasks.sort(key=lambda x: (x["end"], x["start"]))
    n = len(tasks)
    
    # Get key stations and compute shortest paths between them
    key_stations = {s0}
    for task in tasks:
        key_stations.add(task["station"])
    
    graph = build_graph(subway, key_stations)
    distances = {}
    
    # Compute shortest paths from each key station
    for station in key_stations:
        distances[station] = dijkstra(graph, station, key_stations)
    
    def get_distance(u: int, v: int) -> int:
        if u == v:
            return 0
        return distances[u][v]
    
    # Binary search for latest compatible task
    def find_latest_compatible(target_start: int, end_idx: int) -> int:
        left, right = 0, end_idx - 1
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            if tasks[mid]['end'] <= target_start:
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    # DP: for each task, compute optimal schedule ending at that task
    dp = [(0, 0, -1)] * n  # (score, fee, parent_index)
    
    for i in range(n):
        task_i = tasks[i]
        station_i = task_i["station"]
        score_i = task_i["score"]
        start_i = task_i["start"]
        
        # Base case: only task i
        base_fee = get_distance(s0, station_i) + get_distance(station_i, s0)
        dp[i] = (score_i, base_fee, -1)
        
        # Find latest compatible predecessor
        latest_compatible = find_latest_compatible(start_i, i)
        
        # For each compatible predecessor, check if extending gives better result
        for j in range(latest_compatible + 1):
            task_j = tasks[j]
            station_j = task_j["station"]
            
            prev_score, prev_fee, _ = dp[j]
            
            # Calculate new score and fee
            new_score = prev_score + score_i
            new_fee = prev_fee - get_distance(station_j, s0) + get_distance(station_j, station_i) + get_distance(station_i, s0)
            
            # Update if better
            current_score, current_fee, _ = dp[i]
            if (new_score > current_score or 
                (new_score == current_score and new_fee < current_fee)):
                dp[i] = (new_score, new_fee, j)
    
    # Find best overall solution
    max_score = 0
    min_fee = 0
    best_end = -1
    
    for i in range(n):
        score, fee, _ = dp[i]
        if (score > max_score or 
            (score == max_score and fee < min_fee) or
            (max_score == 0 and score > 0)):
            max_score = score
            min_fee = fee
            best_end = i
    
    if max_score == 0:
        return {"max_score": 0, "min_fee": 0, "schedule": []}
    
    # Reconstruct schedule
    schedule = []
    current = best_end
    
    while current != -1:
        schedule.append(tasks[current]["name"])
        _, _, parent = dp[current]
        current = parent
    
    schedule.reverse()
    
    # Sort by start time as required
    task_by_name = {t["name"]: t for t in tasks}
    schedule.sort(key=lambda name: task_by_name[name]["start"])
    
    return {
        "max_score": int(max_score),
        "min_fee": int(min_fee),
        "schedule": schedule
    }