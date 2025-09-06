import math
from collections import defaultdict

def find_profitable_cycles(goods, ratios):
    """
    Find profitable trading cycles using Bellman-Ford algorithm.
    Returns the most profitable cycle path and its gain.
    """
    n = len(goods)
    
    # Build adjacency list from ratios
    # ratios format: [from_index, to_index, rate]
    graph = defaultdict(list)
    for from_idx, to_idx, rate in ratios:
        # Convert to negative log for Bellman-Ford (to find positive cycles)
        weight = -math.log(rate)
        graph[from_idx].append((to_idx, weight, rate))
    
    best_cycle = None
    best_gain = 0
    
    # Try each good as a starting point
    for start in range(n):
        cycle, gain = bellman_ford_detect_cycle(start, n, graph, goods)
        if gain > best_gain:
            best_cycle = cycle
            best_gain = gain
    
    return best_cycle, best_gain

def bellman_ford_detect_cycle(start, n, graph, goods):
    """
    Modified Bellman-Ford to detect positive cycles and reconstruct the path.
    """
    # Initialize distances
    dist = [float('inf')] * n
    pred = [-1] * n
    dist[start] = 0
    
    # Relax edges n-1 times
    for _ in range(n - 1):
        updated = False
        for u in range(n):
            if dist[u] != float('inf'):
                for v, weight, rate in graph[u]:
                    if dist[u] + weight < dist[v]:
                        dist[v] = dist[u] + weight
                        pred[v] = u
                        updated = True
        if not updated:
            break
    
    # Check for negative cycles (which means positive trading cycles)
    cycle_nodes = set()
    for u in range(n):
        if dist[u] != float('inf'):
            for v, weight, rate in graph[u]:
                if dist[u] + weight < dist[v]:
                    cycle_nodes.add(v)
    
    if not cycle_nodes:
        return None, 0
    
    # Find a cycle by following predecessors
    # Start from any node that's part of a negative cycle
    start_node = next(iter(cycle_nodes))
    
    # Follow predecessors to ensure we're in the cycle
    current = start_node
    for _ in range(n):
        if pred[current] != -1:
            current = pred[current]
    
    # Now construct the cycle
    cycle_path = []
    visited = set()
    
    while current not in visited:
        visited.add(current)
        cycle_path.append(current)
        if pred[current] == -1:
            break
        current = pred[current]
    
    # Find where the cycle actually starts
    if current in visited:
        cycle_start_idx = cycle_path.index(current)
        actual_cycle = cycle_path[cycle_start_idx:]
        actual_cycle.append(current)  # Close the cycle
        
        # Calculate the actual gain
        gain = calculate_cycle_gain(actual_cycle, graph)
        return [goods[i] for i in actual_cycle], gain
    
    return None, 0

def calculate_cycle_gain(cycle_indices, graph):
    """
    Calculate the actual gain from a cycle of trades.
    """
    if len(cycle_indices) < 2:
        return 0
    
    total_rate = 1.0
    for i in range(len(cycle_indices) - 1):
        from_idx = cycle_indices[i]
        to_idx = cycle_indices[i + 1]
        
        # Find the rate for this edge
        rate_found = False
        for neighbor, weight, rate in graph[from_idx]:
            if neighbor == to_idx:
                total_rate *= rate
                rate_found = True
                break
        
        if not rate_found:
            return 0  # Invalid cycle
    
    return (total_rate - 1.0) * 100  # Return percentage gain

# Test with the sample data
def solve_ink_archive_challenge():
    # Challenge 1 data
    challenge1 = {
        "ratios": [
            [0, 1, 0.9],
            [1, 2, 120],
            [2, 0, 0.008],
            [0, 3, 0.00005],
            [3, 1, 18000],
            [1, 0, 1.11],
            [2, 3, 4e-7],
            [3, 2, 2600000],
            [1, 3, 0.000055],
            [3, 0, 20000],
            [2, 1, 0.0075]
        ],
        "goods": [
            "Blue Moss",
            "Amberback Shells", 
            "Kelp Silk",
            "Ventspice"
        ]
    }
    
    print("=== Challenge 1: The Whispering Loop ===")
    print("Goods:", challenge1["goods"])
    print("Looking for profitable trading cycles...")
    
    cycle, gain = find_profitable_cycles(challenge1["goods"], challenge1["ratios"])
    
    if cycle:
        print(f"Found profitable cycle: {' -> '.join(cycle)}")
        print(f"Gain: {gain:.2f}%")
    else:
        print("No profitable cycles found")
    
    return cycle, gain

# Run the solution
if __name__ == "__main__":
    solve_ink_archive_challenge()