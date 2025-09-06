import math
from collections import defaultdict
import logging
import json

logger = logging.getLogger(__name__)

def find_profitable_cycles_floyd_warshall(goods, ratios):
    """
    Find the most profitable cycle using Floyd-Warshall algorithm.
    This is more reliable than Bellman-Ford for finding all cycles.
    """
    n = len(goods)
    
    # Initialize distance matrix with infinity
    # Use negative log of rates (so profitable cycles become negative cycles)
    dist = [[float('inf')] * n for _ in range(n)]
    next_node = [[None] * n for _ in range(n)]
    
    # Set diagonal to 0 (no cost to stay at same good)
    for i in range(n):
        dist[i][i] = 0
    
    # Fill in the direct edges
    for from_idx, to_idx, rate in ratios:
        from_idx = int(from_idx)
        to_idx = int(to_idx)
        
        if from_idx >= n or to_idx >= n or from_idx < 0 or to_idx < 0 or rate <= 0:
            continue
            
        weight = -math.log(rate)  # Convert to negative log
        if weight < dist[from_idx][to_idx]:
            dist[from_idx][to_idx] = weight
            next_node[from_idx][to_idx] = to_idx
    
    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]
    
    # Find the most profitable cycle
    best_cycle = None
    best_gain = 0
    
    for i in range(n):
        if dist[i][i] < -1e-9:  # Negative cycle exists (profitable in original terms)
            # Reconstruct the cycle
            cycle_path = reconstruct_cycle(i, next_node)
            if cycle_path:
                cycle_goods = [goods[idx] for idx in cycle_path]
                gain = calculate_cycle_gain_direct(cycle_path, ratios, goods)
                if gain > best_gain:
                    best_cycle = cycle_goods
                    best_gain = gain
    
    return best_cycle, best_gain

def reconstruct_cycle(start, next_node):
    """
    Reconstruct a cycle starting from a given node.
    """
    path = [start]
    current = next_node[start][start]
    
    # Follow the path until we return to start
    while current is not None and current != start and len(path) < len(next_node):
        path.append(current)
        if current == start:
            break
        current = next_node[current][start]
    
    if current == start:
        path.append(start)  # Close the cycle
        return path
    
    return None

def calculate_cycle_gain_direct(cycle_path, ratios, goods):
    """
    Calculate the gain of a cycle using the original ratios.
    """
    if len(cycle_path) < 2:
        return 0
    
    # Build a lookup for rates
    rate_lookup = {}
    for from_idx, to_idx, rate in ratios:
        from_idx = int(from_idx)
        to_idx = int(to_idx)
        rate_lookup[(from_idx, to_idx)] = rate
    
    total_rate = 1.0
    for i in range(len(cycle_path) - 1):
        from_idx = cycle_path[i]
        to_idx = cycle_path[i + 1]
        
        if (from_idx, to_idx) in rate_lookup:
            total_rate *= rate_lookup[(from_idx, to_idx)]
        else:
            return 0  # Invalid cycle
    
    return (total_rate - 1.0) * 100

def find_profitable_cycles_brute_force(goods, ratios):
    """
    Brute force approach: try all possible cycles up to a reasonable length.
    This is guaranteed to find the optimal short cycles.
    """
    n = len(goods)
    
    # Build adjacency list
    graph = defaultdict(list)
    for from_idx, to_idx, rate in ratios:
        from_idx = int(from_idx)
        to_idx = int(to_idx)
        
        if from_idx >= n or to_idx >= n or from_idx < 0 or to_idx < 0 or rate <= 0:
            continue
            
        graph[from_idx].append((to_idx, rate))
    
    best_cycle = None
    best_gain = 0
    
    # Try cycles of different lengths
    for cycle_length in range(2, min(6, n + 1)):  # Cycles of length 2-5
        cycles = find_cycles_of_length(graph, goods, cycle_length)
        for cycle_goods, gain in cycles:
            if gain > best_gain:
                best_cycle = cycle_goods
                best_gain = gain
    
    return best_cycle, best_gain

def find_cycles_of_length(graph, goods, target_length):
    """
    Find all cycles of a specific length using DFS.
    """
    n = len(goods)
    cycles = []
    
    def dfs(start, current, path, total_rate):
        if len(path) == target_length:
            # Check if we can return to start
            for neighbor, rate in graph[current]:
                if neighbor == start:
                    final_rate = total_rate * rate
                    gain = (final_rate - 1.0) * 100
                    if gain > 0:
                        cycle_path = path + [start]
                        cycle_goods = [goods[i] for i in cycle_path]
                        cycles.append((cycle_goods, gain))
            return
        
        if len(path) < target_length:
            for neighbor, rate in graph[current]:
                if neighbor not in path:  # Avoid revisiting nodes
                    dfs(start, neighbor, path + [neighbor], total_rate * rate)
    
    # Try starting from each node
    for start in range(n):
        dfs(start, start, [start], 1.0)
    
    return cycles

def solve_ink_archive_challenge(input_data):
    """
    Solve both parts using multiple approaches and pick the best result.
    """
    results = []
    
    for i, challenge in enumerate(input_data):
        goods = challenge["goods"]
        ratios = challenge["ratios"]
        
        print(f"\n=== Processing Challenge {i+1} ===")
        print(f"Goods: {len(goods)}")
        print(f"Ratios: {len(ratios)}")
        
        # Try multiple approaches and pick the best
        methods = [
            ("Floyd-Warshall", find_profitable_cycles_floyd_warshall),
            ("Brute Force", find_profitable_cycles_brute_force)
        ]
        
        best_cycle = None
        best_gain = 0
        best_method = None
        
        for method_name, method_func in methods:
            try:
                cycle, gain = method_func(goods, ratios)
                print(f"{method_name}: {cycle if cycle else 'No cycle'}, Gain: {gain:.6f}%")
                
                if gain > best_gain:
                    best_cycle = cycle
                    best_gain = gain
                    best_method = method_name
            except Exception as e:
                print(f"{method_name} failed: {e}")
        
        if best_cycle:
            print(f"Best result from {best_method}: {' -> '.join(best_cycle)}")
            print(f"Gain: {best_gain}%")
            
            results.append({
                "path": best_cycle,
                "gain": best_gain
            })
        else:
            print("No profitable cycles found")
            results.append({
                "path": [],
                "gain": 0
            })
    
    return results

# Test data
def test_with_actual_data():
    actual_input = [
        {
            "ratios": [
                [0.0, 1.0, 0.9],
                [1.0, 2.0, 120.0],
                [2.0, 0.0, 0.008],
                [0.0, 3.0, 0.00005],
                [3.0, 1.0, 18000.0],
                [1.0, 0.0, 1.11],
                [2.0, 3.0, 0.0000004],
                [3.0, 2.0, 2600000.0],
                [1.0, 3.0, 0.000055],
                [3.0, 0.0, 20000.0],
                [2.0, 1.0, 0.0075]
            ],
            "goods": [
                "Blue Moss",
                "Amberback Shells",
                "Kelp Silk",
                "Ventspice"
            ]
        },
        {
            "ratios": [
                [0.0, 1.0, 0.9],
                [1.0, 2.0, 1.1],
                [2.0, 0.0, 1.2],
                [1.0, 0.0, 1.1],
                [2.0, 1.0, 0.95],
                [0.0, 8.0, 0.8740536415671486],
                [15.0, 13.0, 0.9050437005117633],
                [19.0, 14.0, 0.883321839947665],
                [13.0, 2.0, 0.9484841540199809],
                [4.0, 15.0, 0.9441249179482114],
                [3.0, 8.0, 0.8628897150873777],
                [12.0, 0.0, 0.8523238122483889],
                [12.0, 3.0, 0.946448686067685],
                [17.0, 7.0, 0.9125146363465559],
                [5.0, 13.0, 0.9276312291274932],
                [0.0, 15.0, 0.8987232847030142],
                [18.0, 11.0, 0.9233152070194993],
                [12.0, 2.0, 0.9338890350047018],
                [16.0, 6.0, 0.9399335011611493],
                [8.0, 17.0, 0.8583062398224914],
                [7.0, 8.0, 0.9222357119188849],
                [11.0, 18.0, 0.8962957818422422],
                [18.0, 12.0, 0.8571498314879894],
                [6.0, 0.0, 0.8838769653535753],
                [15.0, 0.0, 0.9471546988851712],
                [17.0, 9.0, 0.9112581104709868],
                [9.0, 3.0, 0.8717570412209685],
                [3.0, 0.0, 0.8509673497300974],
                [16.0, 3.0, 0.927131296617068],
                [5.0, 11.0, 0.8711235374929895],
                [19.0, 13.0, 0.9445333238959629],
                [11.0, 17.0, 0.8894203552777331],
                [10.0, 5.0, 0.9286042450814552],
                [6.0, 1.0, 0.8670291530247703],
                [11.0, 14.0, 0.9224295033578869],
                [6.0, 4.0, 0.9304395417224636],
                [12.0, 10.0, 0.8962087990285994],
                [9.0, 5.0, 0.9001834850205734],
                [5.0, 2.0, 0.9469269909940415],
                [17.0, 11.0, 0.8547265869196129],
                [17.0, 18.0, 0.8529107512721635],
                [15.0, 1.0, 0.9471950120953745],
                [13.0, 17.0, 0.926744210301549],
                [16.0, 18.0, 0.8755525310896444],
                [15.0, 5.0, 0.9348280500272342],
                [13.0, 10.0, 0.8853852969708712],
                [18.0, 1.0, 0.85036769023557],
                [19.0, 11.0, 0.9031608556248797],
                [17.0, 1.0, 0.8777210026068711],
                [4.0, 3.0, 0.9109881513512763]
            ],
            "goods": [
                "Drift Kelp",
                "Sponge Flesh",
                "Saltbeads",
                "Stoneworm Paste",
                "Mudshrimp",
                "Algae Cakes",
                "Coral Pollen",
                "Rockmilk",
                "Kelp Tubers",
                "Shell Grain",
                "Inkbinding Resin",
                "Reef Clay",
                "Dry Coral Fiber",
                "Scale Sand",
                "Glowmoss Threads",
                "Filter Sponges",
                "Bubble Nets",
                "Siltstone Tools",
                "Shell Chits",
                "Crush Coral Blocks"
            ]
        }
    ]
    
    return solve_ink_archive_challenge(actual_input)

if __name__ == "__main__":
    print("=== Testing Multiple Approaches ===")
    results = test_with_actual_data()
    
    print(f"\n=== Final JSON Output ===")
    print(json.dumps(results, indent=2))