"""
Lab 6 - Task 4: Weighted A* for an Autonomous Delivery Drone
Run with:  python task4_weighted_astar.py
Requires:  pip install networkx matplotlib
"""

import math
import heapq
import networkx as nx
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Coordinates
# ------------------------------------------------------------------
locations = {
    "Distribution_Center": (0, 0),
    "Zone_A": (2, 1),
    "Zone_B": (1, 4),
    "Zone_C": (4, 2),
    "Zone_D": (5, 5),
    "Customer_Building": (8, 6),
}

# ------------------------------------------------------------------
# 2. Weighted graph (edge costs read from the lab figure;
#    note Distribution_Center -> Zone_B is 3.0 in this task's figure)
# ------------------------------------------------------------------
drone_graph = {
    "Distribution_Center": {"Zone_A": 2.2, "Zone_B": 3.0},
    "Zone_A": {"Zone_C": 2.2},
    "Zone_B": {"Zone_D": 5.0},
    "Zone_C": {"Zone_D": 3.2, "Customer_Building": 6.0},
    "Zone_D": {"Customer_Building": 3.2},
    "Customer_Building": {},
}


# ------------------------------------------------------------------
# 3. Euclidean heuristic
# ------------------------------------------------------------------
def heuristic(current, goal):
    x1, y1 = locations[current]
    x2, y2 = locations[goal]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def reconstruct_path(came_from, current):
    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


# ------------------------------------------------------------------
# 4. Weighted A*:  f(n) = g(n) + w * h(n)      (w = 1 is normal A*)
# ------------------------------------------------------------------
def weighted_a_star(start, goal, graph, w=1.0):
    g_cost = {start: 0.0}
    came_from = {start: None}
    frontier = [(w * heuristic(start, goal), start)]
    expanded = set()
    expansion_order = []

    while frontier:
        _, current = heapq.heappop(frontier)
        if current in expanded:
            continue
        expanded.add(current)
        expansion_order.append(current)

        if current == goal:
            return reconstruct_path(came_from, current), g_cost[goal], expansion_order

        for neighbor, cost in graph[current].items():
            tentative_g = g_cost[current] + cost
            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g
                came_from[neighbor] = current
                f = tentative_g + w * heuristic(neighbor, goal)
                heapq.heappush(frontier, (f, neighbor))

    return None, None, expansion_order


# ------------------------------------------------------------------
# 5. Run for w = 1, 1.5, 2, 3
# ------------------------------------------------------------------
start = "Distribution_Center"
goal = "Customer_Building"
weights = [1, 1.5, 2, 3]

print("Heuristic values (Euclidean distance to Customer_Building)")
print("-" * 55)
for node in locations:
    print(f"{node:<22}{heuristic(node, goal):>8.2f}")

results = {}
for w in weights:
    path, cost, order = weighted_a_star(start, goal, drone_graph, w)
    results[w] = (path, cost, order)

print("\nExpansion order for each w")
print("-" * 55)
for w in weights:
    print(f"w = {w:<4}: {' -> '.join(results[w][2])}")

print("\nResults table")
print("=" * 118)
print(f"{'Weight w':<10}{'Solution Path':<86}{'Total Cost':<12}{'Nodes Expanded'}")
print("-" * 118)
for w in weights:
    path, cost, order = results[w]
    print(f"{w:<10}{' -> '.join(path):<86}{cost:<12.2f}{len(order)}")
print("=" * 118)

# ------------------------------------------------------------------
# 6. Visualization: one subplot per weight
# ------------------------------------------------------------------
G = nx.DiGraph()
for node, neighbors in drone_graph.items():
    G.add_node(node)
    for neighbor, weight in neighbors.items():
        G.add_edge(node, neighbor, weight=weight)

pos = locations
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

for ax, w in zip(axes.flatten(), weights):
    path, cost, order = results[w]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1500, node_color="lightblue", edgecolors="black")
    nx.draw_networkx_edges(G, pos, ax=ax, node_size=1500, arrowsize=15, edge_color="gray")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_weight="bold")
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=nx.get_edge_attributes(G, "weight"), font_size=8)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=path, node_size=1500, node_color="orange", edgecolors="black")
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=list(zip(path, path[1:])), node_size=1500,
                           edge_color="red", width=3, arrowsize=20)
    ax.set_title(f"w = {w}  |  cost = {cost:.2f}  |  nodes expanded = {len(order)}")
    ax.axis("off")

plt.suptitle("Weighted A* - Autonomous Delivery Drone", fontsize=14)
plt.tight_layout()
plt.savefig("task4_weighted_astar_graphs.png", dpi=150, bbox_inches="tight")
plt.show()