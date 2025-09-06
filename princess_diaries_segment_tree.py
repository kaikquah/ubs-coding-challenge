import heapq
import bisect
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

INF = float('inf')

class SegmentTree:
    """Segment tree for range maximum queries with (score, -fee) tuples"""
    
    def __init__(self, n: int):
        self.n = n
        self.tree = [(0, 0, -1)] * (4 * n)  # (score, fee, index)
    
    def update(self, pos: int, score: int, fee: int, idx: int):
        """Update position pos with (score, fee, idx)"""
        self._update(1, 0, self.n - 1, pos, score, fee, idx)
    
    def _update(self, node: int, start: int, end: int, pos: int, score: int, fee: int, idx: int):
        if start == end:
            # Update if better score or same score with lower fee
            current_score, current_fee, _ = self.tree[node]
            if score > current_score or (score == current_score and fee < current_fee):
                self.tree[node] = (score, fee, idx)
        else:
            mid = (start + end) // 2
            if pos <= mid:
                self._update(2 * node, start, mid, pos, score, fee, idx)
            else:
                self._update(2 * node + 1, mid + 1, end, pos, score, fee, idx)
            
            # Update current node with best of children
            left_score, left_fee, left_idx = self.tree[2 * node]
            right_score, right_fee, right_idx = self.tree[2 * node + 1]
            
            if (left_score > right_score or 
                (left_score == right_score and left_fee < right_fee)):
                self.tree[node] = (left_score, left_fee, left_idx)
            else:
                self.tree[node] = (right_score, right_fee, right_idx)
    
    def query(self, left: int, right: int) -> Tuple[int, int, int]:
        """Query for best (score, fee, index) in range [left, right]"""
        if left > right:
            return (0, 0, -1)
        return self._query(1, 0, self.n - 1, left, right)
    
    def _query(self, node: int, start: int, end: int, left: int, right: int) -> Tuple[int, int, int]:
        if right < start or left > end:
            return (0, 0, -1)
        
        if left <= start and end <= right:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_result = self._query(2 * node, start, mid, left, right)
        right_result = self._query(2 * node + 1, mid + 1, end, left, right)
        
        left_score, left_fee, left_idx = left_result
        right_score, right_fee, right_idx = right_result
        
        if left_idx == -1:
            return right_result
        if right_idx == -1:
            return left_result
        
        if (left_score > right_score or 
            (left_score == right_score and left_fee < right_fee)):
            return left_result
        else:
            return right_result

def dijkstra(graph: Dict[int, List[Tuple[int, int]]], start: int, targets: set) -> Dict[int, int]:
    """Dijkstra's algorithm optimized to stop early when all targets found"""
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
    """Build adjacency list representation of the subway graph"""
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
    
    # Sort tasks by end time, then by start time for stability
    tasks.sort(key=lambda x: (x["end"], x["start"]))
    n = len(tasks)
    
    # Get all key stations and compute shortest paths
    key_stations = {s0}
    for task in tasks:
        key_stations.add(task["station"])
    
    graph = build_graph(subway, key_stations)
    distances = {}
    
    for station in key_stations:
        distances[station] = dijkstra(graph, station, key_stations)
    
    def get_distance(u: int, v: int) -> int:
        if u == v:
            return 0
        return distances[u][v]
    
    # DP with segment tree for efficient predecessor queries
    dp = [(0, 0, -1)] * n  # (score, fee, parent)
    seg_tree = SegmentTree(n)
    
    # Process tasks in order
    for i in range(n):
        task_i = tasks[i]
        station_i = task_i["station"]
        score_i = task_i["score"]
        start_i = task_i["start"]
        
        # Base case: only do task i
        base_fee = get_distance(s0, station_i) + get_distance(station_i, s0)
        dp[i] = (score_i, base_fee, -1)
        
        # Find best predecessor using binary search + segment tree
        # Binary search for latest task that ends before start_i
        left, right = 0, i - 1
        latest_compatible = -1
        
        while left <= right:
            mid = (left + right) // 2
            if tasks[mid]["end"] <= start_i:
                latest_compatible = mid
                left = mid + 1
            else:
                right = mid - 1
        
        if latest_compatible >= 0:
            # Query segment tree for best predecessor in range [0, latest_compatible]
            best_score, best_fee, best_idx = seg_tree.query(0, latest_compatible)
            
            if best_idx != -1:
                task_j = tasks[best_idx]
                station_j = task_j["station"]
                
                # Calculate score and fee if extending from best predecessor
                extend_score = best_score + score_i
                extend_fee = best_fee - get_distance(station_j, s0) + get_distance(station_j, station_i) + get_distance(station_i, s0)
                
                # Update if better
                if (extend_score > dp[i][0] or 
                    (extend_score == dp[i][0] and extend_fee < dp[i][1])):
                    dp[i] = (extend_score, extend_fee, best_idx)
        
        # Update segment tree with current solution
        seg_tree.update(i, dp[i][0], dp[i][1], i)
    
    # Find overall best solution
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
        current = dp[current][2]  # parent index
    
    schedule.reverse()
    
    # Sort by start time as required
    task_by_name = {t["name"]: t for t in tasks}
    schedule.sort(key=lambda name: task_by_name[name]["start"])
    
    return {
        "max_score": int(max_score),
        "min_fee": int(min_fee),
        "schedule": schedule
    }