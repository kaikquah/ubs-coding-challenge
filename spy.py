import json

def buildAdjacencyList(edges):
    """Build adjacency list from edge list"""
    adjList = {}
    for edge in edges:
        spy1, spy2 = edge["spy1"], edge["spy2"]
        if spy1 not in adjList:
            adjList[spy1] = []
        if spy2 not in adjList:
            adjList[spy2] = []
        adjList[spy1].append(spy2)
        adjList[spy2].append(spy1)
    return adjList

def processNetwork(edges):
    """
    Find all edges that participate in cycles.
    
    Algorithm:
    1. Build adjacency list from edges
    2. Use DFS to find all cycles by tracking paths
    3. When a back edge is found, trace back to find the complete cycle
    4. Mark all edges in each cycle as participating in cycles
    """
    if not edges:
        return []
    
    adjList = buildAdjacencyList(edges)
    visited = set()
    cycle_edges = set()
    
    # Create edge lookup for easy identification
    edge_set = set()
    for edge in edges:
        spy1, spy2 = edge["spy1"], edge["spy2"]
        edge_key = (min(spy1, spy2), max(spy1, spy2))
        edge_set.add(edge_key)
    
    def dfs(node, parent, path):
        visited.add(node)
        path.append(node)
        
        for neighbor in adjList[node]:
            if neighbor == parent:
                continue  # Skip the edge we came from
            
            if neighbor in path:
                # Found a cycle - mark all edges in the cycle
                cycle_start_idx = path.index(neighbor)
                cycle_path = path[cycle_start_idx:] + [neighbor]
                
                # Add all edges in this cycle to our result set
                for i in range(len(cycle_path) - 1):
                    edge_key = (min(cycle_path[i], cycle_path[i + 1]), 
                               max(cycle_path[i], cycle_path[i + 1]))
                    if edge_key in edge_set:  # Ensure edge exists in original graph
                        cycle_edges.add(edge_key)
            
            elif neighbor not in visited:
                dfs(neighbor, node, path)
        
        path.pop()  # Remove from path when backtracking
    
    # Run DFS from each unvisited node
    for node in adjList:
        if node not in visited:
            dfs(node, None, [])
    
    # Convert tuples back to required dictionary format
    result = []
    for edge in cycle_edges:
        result.append({"spy1": edge[0], "spy2": edge[1]})
    
    return result

def process_data(data):
    """Main function to process all networks and find extra channels"""
    result = {'networks': []}
    
    for network in data['networks']:
        networkData = {
            'networkId': network['networkId'],
            'extraChannels': processNetwork(network['network'])
        }
        result['networks'].append(networkData)
    
    return result

# Test with sample data
if __name__ == "__main__":
    sampleJsonString = '''
    {
      "networks": [
        {
          "networkId": "network1",
          "network": [
            {
              "spy1": "Karina",
              "spy2": "Giselle"
            },
            {
              "spy1": "Karina",
              "spy2": "Winter"
            },
            {
              "spy1": "Karina",
              "spy2": "Ningning"
            },
            {
              "spy1": "Giselle",
              "spy2": "Winter"
            }
          ]
        }
      ]
    }
    '''
    
    # Test with a more complex example that has multiple cycles
    complexJsonString = '''
    {
      "networks": [
        {
          "networkId": "network1",
          "network": [
            {
              "spy1": "A",
              "spy2": "B"
            },
            {
              "spy1": "B",
              "spy2": "C"
            },
            {
              "spy1": "C",
              "spy2": "A"
            },
            {
              "spy1": "C",
              "spy2": "D"
            },
            {
              "spy1": "D",
              "spy2": "E"
            },
            {
              "spy1": "E",
              "spy2": "C"
            }
          ]
        }
      ]
    }
    '''
    
    print("Original sample:")
    sampleinput = json.loads(sampleJsonString)
    result = process_data(sampleinput)
    print(json.dumps(result, indent=2))
    
    print("\nComplex example with multiple cycles:")
    complexinput = json.loads(complexJsonString)
    result2 = process_data(complexinput)
    print(json.dumps(result2, indent=2))