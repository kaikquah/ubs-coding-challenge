import heapq
from collections import defaultdict

INF = float('inf')

# Precompute all-pairs shortest paths using Floyd-Warshall
def floyd_warshall(n, graph):
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u in range(n):
        for v, fee in graph[u]:
            dist[u][v] = fee
            dist[v][u] = fee
    # Floyd-Warshall Algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

def solve_princess_diaries(payload):
    tasks = payload.get("tasks", [])
    subway = payload.get("subway", [])
    s0 = payload.get("starting_station")

    # Build the graph for subway system
    graph = defaultdict(list)
    stations = set([s0])
    for route in subway:
        u, v = route["connection"]
        w = route["fee"]
        graph[u].append((v, w))
        graph[v].append((u, w))
        stations.add(u)
        stations.add(v)

    # Number of stations
    n = len(stations)

    # Compute all-pairs shortest paths using Floyd-Warshall
    dist = floyd_warshall(n, graph)

    # Sort tasks by their start time
    tasks.sort(key=lambda x: x["start"])

    # DP approach: dp[i] = (best_score, best_fee)
    dp = [(0, 0)] * len(tasks)  # (score, fee)
    prev_task = [-1] * len(tasks)  # For reconstructing schedule

    # Process tasks
    for i in range(len(tasks)):
        task_i = tasks[i]
        start_i = task_i["start"]
        end_i = task_i["end"]
        station_i = task_i["station"]
        score_i = task_i["score"]
        
        # Initial case: Task i on its own
        dp[i] = (score_i, dist[s0][station_i] + dist[station_i][s0])

        # Look for previous tasks that can be done before task i
        for j in range(i):
            task_j = tasks[j]
            end_j = task_j["end"]
            start_j = task_j["start"]
            station_j = task_j["station"]

            if end_j <= start_i:  # Tasks do not overlap
                prev_score, prev_fee = dp[j]
                cost = prev_fee - dist[station_j][s0] + dist[station_j][station_i] + dist[station_i][s0]
                new_score = prev_score + score_i

                # Update if we get better score or lower fee with same score
                if new_score > dp[i][0] or (new_score == dp[i][0] and cost < dp[i][1]):
                    dp[i] = (new_score, cost)
                    prev_task[i] = j

    # Find the best possible score and fee
    max_score, min_fee = max(dp, key=lambda x: (x[0], -x[1]))

    # Reconstruct the schedule
    schedule = []
    idx = dp.index((max_score, min_fee))
    while idx != -1:
        schedule.append(tasks[idx]["name"])
        idx = prev_task[idx]

    schedule.reverse()

    return {
        "max_score": max_score,
        "min_fee": min_fee,
        "schedule": schedule
    }