"""
Lab 6 - Task 2: Greedy Best-First Search for Airport Baggage Handling
Run with:  python task2_gbfs_airport.py
Requires:  pip install networkx matplotlib
"""

import math
import heapq
import networkx as nx
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Airport coordinates
# ------------------------------------------------------------------
locations = {
    "Baggage_Area": (0, 0),
    "Security": (2, 1),
    "Checkpoint": (1, 4),
    "Food_Court": (4, 2),
    "Terminal_Hall": (5, 5),
    "Departure_Gate": (8, 6),
}

# ------------------------------------------------------------------
# 2. Weighted airport graph
# ------------------------------------------------------------------
airport_graph = {
    "Baggage_Area": {"Security": 2.2, "Checkpoint": 4.1},
    "Security": {"Food_Court": 2.2},
    "Checkpoint": {"Terminal_Hall": 5.0},
    "Food_Court": {"Terminal_Hall": 3.2, "Departure_Gate": 6.0},
    "Terminal_Hall": {"Departure_Gate": 3.2},
    "Departure_Gate": {},
}


# ------------------------------------------------------------------
# 3. Euclidean heuristic
# ------------------------------------------------------------------
def heuristic(current, goal):
    x1, y1 = locations[current]
    x2, y2 = locations[goal]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# ------------------------------------------------------------------
# 4. Greedy Best-First Search  (priority = h(n) only)
# ------------------------------------------------------------------
def reconstruct_path(came_from, current):
    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


def greedy_best_first_search(start, goal, graph):
    frontier = [(heuristic(start, goal), start)]   # min-heap ordered by h(n)
    came_from = {start: None}
    explored = set()
    expansion_order = []

    while frontier:
        h_value, current = heapq.heappop(frontier)

        if current in explored:
            continue
        explored.add(current)
        expansion_order.append(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            total_cost = sum(graph[a][b] for a, b in zip(path, path[1:]))
            return expansion_order, path, total_cost

        for neighbor in graph[current]:
            if neighbor not in explored:
                if neighbor not in came_from:
                    came_from[neighbor] = current
                heapq.heappush(frontier, (heuristic(neighbor, goal), neighbor))

    return expansion_order, None, None   # failure


# ------------------------------------------------------------------
# 5. Run GBFS
# ------------------------------------------------------------------
start = "Baggage_Area"
goal = "Departure_Gate"

print("Heuristic values (Euclidean distance to Departure_Gate)")
print("--------------------------------------------------------")
for node in locations:
    print(f"{node:<16}{heuristic(node, goal):>8.2f}")

expansion_order, path, total_cost = greedy_best_first_search(start, goal, airport_graph)

print("\nGBFS")
print("--------------------------------")
print("Expansion order:")
print(" -> ".join(expansion_order))

if path is None:
    print("\nNo path found.")
    raise SystemExit

print("\nSolution path:")
print(" -> ".join(path))
print(f"\nTotal path cost: {total_cost:.2f}")

# Optimal cost for comparison (reference only)
G = nx.DiGraph()
for node, neighbors in airport_graph.items():
    G.add_node(node)
    for neighbor, weight in neighbors.items():
        G.add_edge(node, neighbor, weight=weight)

optimal_path = nx.shortest_path(G, start, goal, weight="weight")
optimal_cost = nx.shortest_path_length(G, start, goal, weight="weight")
print("\nFor comparison, the optimal path is:")
print(" -> ".join(optimal_path), f"(cost {optimal_cost:.2f})")

print(
    "\nExplanation: From Baggage_Area, GBFS compares neighbors by h(n) only. "
    "Checkpoint (h=7.28) looks closer to the gate than Security (h=7.81), so it is "
    "expanded first. Its only neighbor Terminal_Hall (h=3.16) is expanded next, and "
    "then Departure_Gate (h=0) is reached. GBFS is fast because the heuristic pulls "
    "it straight toward the goal, but since it ignores g(n) the path costs "
    f"{total_cost:.1f}, higher than the optimal {optimal_cost:.1f}."
)

# ------------------------------------------------------------------
# 6. NetworkX visualization with highlighted solution path
# ------------------------------------------------------------------
pos = locations

plt.figure(figsize=(12, 7))

nx.draw_networkx_nodes(G, pos, node_size=2200, node_color="lightblue", edgecolors="black")
nx.draw_networkx_edges(G, pos, node_size=2200, arrowsize=20, edge_color="gray")
nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
nx.draw_networkx_edge_labels(
    G, pos, edge_labels=nx.get_edge_attributes(G, "weight"), font_size=9
)

path_edges = list(zip(path, path[1:]))
nx.draw_networkx_nodes(
    G, pos, nodelist=path, node_size=2200, node_color="orange", edgecolors="black"
)
nx.draw_networkx_edges(
    G, pos, edgelist=path_edges, node_size=2200, edge_color="red", width=3, arrowsize=25
)

plt.title("Airport Baggage Handling - GBFS")
plt.axis("off")
plt.savefig("task2_gbfs_graph.png", dpi=150, bbox_inches="tight")
plt.show()