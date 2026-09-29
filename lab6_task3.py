"""
Lab 6 - Task 3: A* Search for an Emergency Supply Robot
Run with:  python task3_astar_hospital.py
Requires:  pip install networkx matplotlib
"""

import math
import heapq
import networkx as nx
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Hospital coordinates
# ------------------------------------------------------------------
locations = {
    "Pharmacy": (0, 0),
    "Main_Corridor": (2, 1),
    "Patient_Wing": (1, 4),
    "Nursing_Station": (4, 2),
    "Laboratory": (5, 5),
    "Emergency_Ward": (8, 6),
}

# ------------------------------------------------------------------
# 2. Hospital weighted graph
# ------------------------------------------------------------------
hospital_graph = {
    "Pharmacy": {"Main_Corridor": 2.2, "Patient_Wing": 4.1},
    "Main_Corridor": {"Nursing_Station": 2.2},
    "Patient_Wing": {"Laboratory": 5.0},
    "Nursing_Station": {"Laboratory": 3.2, "Emergency_Ward": 6.0},
    "Laboratory": {"Emergency_Ward": 3.2},
    "Emergency_Ward": {},
}


# ------------------------------------------------------------------
# 3. Euclidean heuristic
# ------------------------------------------------------------------
def heuristic(current, goal):
    x1, y1 = locations[current]
    x2, y2 = locations[goal]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# ------------------------------------------------------------------
# 4. Path reconstruction
# ------------------------------------------------------------------
def reconstruct_path(came_from, current):
    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


# ------------------------------------------------------------------
# 5. A* Search  (priority = f(n) = g(n) + h(n))
# ------------------------------------------------------------------
def a_star_search(start, goal, graph):
    g_cost = {start: 0.0}
    came_from = {start: None}
    frontier = [(heuristic(start, goal), start)]   # (f, node)
    expanded = set()
    expansion_order = []
    expansion_details = []                          # (node, g, h, f) when expanded

    while frontier:
        f_value, current = heapq.heappop(frontier)

        # skip stale queue entries (a cheaper route was found later)
        if current in expanded:
            continue
        expanded.add(current)

        g = g_cost[current]
        h = heuristic(current, goal)
        expansion_order.append(current)
        expansion_details.append((current, g, h, g + h))

        if current == goal:
            path = reconstruct_path(came_from, current)
            return expansion_order, expansion_details, path, g_cost[goal]

        for neighbor, cost in graph[current].items():
            tentative_g = g_cost[current] + cost
            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g
                came_from[neighbor] = current
                f_neighbor = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(frontier, (f_neighbor, neighbor))

    return expansion_order, expansion_details, None, None   # failure


# ------------------------------------------------------------------
# GBFS (only used for the comparison at the end)
# ------------------------------------------------------------------
def greedy_best_first_search(start, goal, graph):
    frontier = [(heuristic(start, goal), start)]
    came_from = {start: None}
    explored = set()
    order = []
    while frontier:
        _, current = heapq.heappop(frontier)
        if current in explored:
            continue
        explored.add(current)
        order.append(current)
        if current == goal:
            path = reconstruct_path(came_from, current)
            cost = sum(graph[a][b] for a, b in zip(path, path[1:]))
            return order, path, cost
        for neighbor in graph[current]:
            if neighbor not in explored:
                if neighbor not in came_from:
                    came_from[neighbor] = current
                heapq.heappush(frontier, (heuristic(neighbor, goal), neighbor))
    return order, None, None


# ------------------------------------------------------------------
# 6. Run A*
# ------------------------------------------------------------------
start = "Pharmacy"
goal = "Emergency_Ward"

expansion_order, details, path, total_cost = a_star_search(start, goal, hospital_graph)

print("A* SEARCH")
print("--------------------------------")
print("Expansion order:")
print(" -> ".join(expansion_order))

if path is None:
    print("\nNo path found.")
    raise SystemExit

print("\nSolution path:")
print(" -> ".join(path))
print(f"\nTotal path cost: {total_cost:.2f}")

print("\ng(n), h(n), f(n) for each expanded node")
print(f"{'Node':<18}{'g(n)':>8}{'h(n)':>8}{'f(n)':>8}")
print("-" * 42)
for node, g, h, f in details:
    print(f"{node:<18}{g:>8.2f}{h:>8.2f}{f:>8.2f}")

# ------------------------------------------------------------------
# 7. Comparison with GBFS
# ------------------------------------------------------------------
gbfs_order, gbfs_path, gbfs_cost = greedy_best_first_search(start, goal, hospital_graph)

print("\nComparison: GBFS vs A*")
print("--------------------------------")
print(f"{'Measure':<22}{'GBFS':<70}{'A*'}")
print(f"{'Expansion order':<22}{' -> '.join(gbfs_order):<70}{' -> '.join(expansion_order)}")
print(f"{'Solution path':<22}{' -> '.join(gbfs_path):<70}{' -> '.join(path)}")
print(f"{'Total cost':<22}{gbfs_cost:<70.2f}{total_cost:.2f}")
print(f"{'Nodes expanded':<22}{len(gbfs_order):<70}{len(expansion_order)}")

print(
    "\nExplanation: GBFS looks only at h(n). From Pharmacy it prefers Patient_Wing "
    "(h=7.28) over Main_Corridor (h=7.81), and ends with a path costing "
    f"{gbfs_cost:.1f}. A* adds g(n): Main_Corridor has f = 2.20 + 7.81 = 10.01 versus "
    "Patient_Wing f = 4.10 + 7.28 = 11.38, so A* explores the cheaper side first, "
    "reaches Emergency_Ward with f = 10.40 before any other frontier node can beat "
    f"it, and returns the optimal path costing {total_cost:.1f}."
)

# ------------------------------------------------------------------
# 8. NetworkX visualization with highlighted solution path
# ------------------------------------------------------------------
G = nx.DiGraph()
for node, neighbors in hospital_graph.items():
    G.add_node(node)
    for neighbor, weight in neighbors.items():
        G.add_edge(node, neighbor, weight=weight)

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

plt.title("Emergency Supply Robot - A* Solution Path")
plt.axis("off")
plt.savefig("task3_astar_graph.png", dpi=150, bbox_inches="tight")
plt.show()