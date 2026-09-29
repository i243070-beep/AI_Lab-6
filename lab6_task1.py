"""
Lab 6 - Task 1: Designing a Heuristic for a Warehouse Robot
Run with:  python task1_warehouse_heuristic.py
Requires:  pip install networkx matplotlib
"""

import math
import networkx as nx
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Coordinates of warehouse locations
# ------------------------------------------------------------------
locations = {
    "Receiving_Area": (0, 0),
    "Storage_A": (2, 1),
    "Storage_B": (1, 4),
    "Sorting_Area": (4, 2),
    "Inspection_Area": (5, 5),
    "Packing_Station": (7, 6),
}

# ------------------------------------------------------------------
# 2. Weighted warehouse graph
# ------------------------------------------------------------------
warehouse_graph = {
    "Receiving_Area": {"Storage_A": 2.2, "Storage_B": 4.1},
    "Storage_A": {"Sorting_Area": 2.2},
    "Storage_B": {"Inspection_Area": 5.0, "Sorting_Area": 6.0},
    "Sorting_Area": {"Inspection_Area": 3.2, "Packing_Station": 5.0},
    "Inspection_Area": {"Packing_Station": 2.2},
    "Packing_Station": {},
}


# ------------------------------------------------------------------
# 3. Euclidean heuristic
# ------------------------------------------------------------------
def heuristic(current, goal):
    x1, y1 = locations[current]
    x2, y2 = locations[goal]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# ------------------------------------------------------------------
# 4. Calculate and display heuristic values
# ------------------------------------------------------------------
goal = "Packing_Station"

print("Heuristic Values (Euclidean distance to Packing_Station)")
print("--------------------------------------------------------")
print(f"{'Location':<20}{'Coordinates':<15}{'h(n)':>8}")
print("-" * 43)
for location in locations:
    h = heuristic(location, goal)
    print(f"{location:<20}{str(locations[location]):<15}{h:>8.2f}")

# ------------------------------------------------------------------
# 5. Create NetworkX graph
# ------------------------------------------------------------------
G = nx.DiGraph()
for node, neighbors in warehouse_graph.items():
    G.add_node(node)
    for neighbor, weight in neighbors.items():
        G.add_edge(node, neighbor, weight=weight)

# ------------------------------------------------------------------
# 6. Minimum-cost path (used only for visualization)
# ------------------------------------------------------------------
solution_path = nx.shortest_path(
    G, source="Receiving_Area", target="Packing_Station", weight="weight"
)
solution_cost = nx.shortest_path_length(
    G, source="Receiving_Area", target="Packing_Station", weight="weight"
)

print("\nMinimum cost path:")
print(" -> ".join(solution_path))
print(f"Total cost: {solution_cost:.2f}")

# ------------------------------------------------------------------
# 7. Verify heuristic condition: h(n) <= cost(n,m) + h(m)
# ------------------------------------------------------------------
print("\nHeuristic Condition Verification")
print("***********************************")

violations = []
for node in warehouse_graph:
    for neighbor, cost in warehouse_graph[node].items():
        h_n = heuristic(node, goal)
        h_m = heuristic(neighbor, goal)
        rhs = cost + h_m
        satisfied = h_n <= rhs
        status = "SATISFIED" if satisfied else "VIOLATED"
        print(
            f"{node:<16} -> {neighbor:<16} "
            f"h(n)={h_n:.3f}  cost+h(m)={cost}+{h_m:.3f}={rhs:.3f}  {status}"
        )
        if not satisfied:
            violations.append((node, neighbor))

print("\nEdges violating the condition:", violations if violations else "None")

print(
    "\nObservation: The condition holds for 7 of the 8 edges. The only exception is\n"
    "Inspection_Area -> Packing_Station, where h(n) = 2.236 is marginally larger than\n"
    "the edge cost 2.2 (the given cost is a rounded value of sqrt(5))."
)

print(
    "\nWhy Euclidean distance is suitable: every location has (x, y) coordinates, a\n"
    "straight line is the shortest possible distance between two points (so it does\n"
    "not exceed the real travel distance, apart from the rounding noted above), and\n"
    "it is cheap to compute while guiding the search toward the Packing Station."
)

# ------------------------------------------------------------------
# 8. Visualize graph with highlighted solution path
# ------------------------------------------------------------------
pos = locations

plt.figure(figsize=(12, 7))

nx.draw_networkx_nodes(G, pos, node_size=2200, node_color="lightblue", edgecolors="black")
nx.draw_networkx_edges(G, pos, node_size=2200, arrowsize=20, edge_color="gray")
nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
nx.draw_networkx_edge_labels(
    G, pos, edge_labels=nx.get_edge_attributes(G, "weight"), font_size=9
)

path_edges = list(zip(solution_path, solution_path[1:]))
nx.draw_networkx_nodes(
    G, pos, nodelist=solution_path, node_size=2200, node_color="orange", edgecolors="black"
)
nx.draw_networkx_edges(
    G, pos, edgelist=path_edges, node_size=2200, edge_color="red", width=3, arrowsize=25
)

plt.title("Warehouse Robot: Weighted Graph (Minimum-Cost Path Highlighted)")
plt.axis("off")
plt.savefig("task1_warehouse_graph.png", dpi=150, bbox_inches="tight")
plt.show()