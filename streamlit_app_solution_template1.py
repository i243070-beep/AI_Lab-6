import streamlit as st
import math
import heapq
import networkx as nx
import matplotlib.pyplot as plt

locations = {
    "Pharmacy": (0, 0),
    "Main_Corridor": (2, 1),
    "Patient_Wing": (1, 4),
    "Nursing_Station": (4, 2),
    "Laboratory": (5, 5),
    "Emergency_Ward": (8, 6)
}

hospital_graph = {
    "Pharmacy": {"Main_Corridor": 2.2, "Patient_Wing": 4.1},
    "Main_Corridor": {"Nursing_Station": 2.2},
    "Patient_Wing": {"Laboratory": 5.0},
    "Nursing_Station": {"Laboratory": 3.2, "Emergency_Ward": 6.0},
    "Laboratory": {"Emergency_Ward": 3.2},
    "Emergency_Ward": {}
}


def heuristic(current, goal):
    x1, y1 = locations[current]
    x2, y2 = locations[goal]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def reconstruct_path(came_from, current):
    path = []
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    path.reverse()
    return path


def gbfs(start, goal):
    frontier = []
    heapq.heappush(frontier, (heuristic(start, goal), start))
    came_from = {start: None}
    explored = set()

    while frontier:
        _, current = heapq.heappop(frontier)
        if current in explored:
            continue
        explored.add(current)
        if current == goal:
            break
        for neighbor in hospital_graph[current]:
            if neighbor not in explored:
                if neighbor not in came_from:
                    came_from[neighbor] = current
                heapq.heappush(frontier, (heuristic(neighbor, goal), neighbor))

    if goal not in came_from:
        return None, None

    path = reconstruct_path(came_from, goal)
    total_cost = sum(hospital_graph[path[i]][path[i + 1]] for i in range(len(path) - 1))
    return path, total_cost


def a_star(start, goal):
    g_cost = {start: 0}
    came_from = {start: None}
    explored = set()
    frontier = []
    heapq.heappush(frontier, (heuristic(start, goal), start))

    while frontier:
        _, current = heapq.heappop(frontier)
        if current in explored:
            continue
        explored.add(current)
        if current == goal:
            break
        for neighbor, cost in hospital_graph[current].items():
            new_g = g_cost[current] + cost
            if neighbor not in g_cost or new_g < g_cost[neighbor]:
                g_cost[neighbor] = new_g
                came_from[neighbor] = current
                heapq.heappush(frontier, (new_g + heuristic(neighbor, goal), neighbor))

    if goal not in came_from:
        return None, None

    path = reconstruct_path(came_from, goal)
    return path, g_cost[goal]


# ─── Streamlit GUI ───────────────────────────────────────────

st.set_page_config(page_title="Hospital Search Visualization", layout="wide")

st.title("Hospital Emergency Supply Robot")
st.markdown(r"Visualize **GBFS** and **A\*** searching for a path through hospital corridors.")

nodes = list(hospital_graph.keys())

col1, col2, col3 = st.columns(3)

with col1:
    start = st.selectbox("Select Initial Node", nodes, index=nodes.index("Pharmacy"))

with col2:
    goal = st.selectbox("Select Goal Node", nodes, index=nodes.index("Emergency_Ward"))

with col3:
    algorithm = st.selectbox("Select Algorithm", ["GBFS", "A*"])

if st.button("Run Search", type="primary"):

    if start == goal:
        st.warning("Start and goal nodes are the same. Please select different nodes.")
    else:
        if algorithm == "GBFS":
            path, cost = gbfs(start, goal)
        else:
            path, cost = a_star(start, goal)

        if path is None:
            st.error(f"No path found from {start} to {goal}.")
        else:
            st.subheader("Search Result")
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Algorithm", algorithm)
            res_col2.metric("Total Path Cost", f"{cost:.2f}")
            res_col3.metric("Path Length", f"{len(path)} nodes")

            st.write(f"**Solution Path:** {' → '.join(path)}")

            G = nx.DiGraph()
            for node, neighbors in hospital_graph.items():
                for neighbor, weight in neighbors.items():
                    G.add_edge(node, neighbor, weight=weight)

            pos = locations
            solution_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            other_edges = [e for e in G.edges() if e not in solution_edges]

            node_colors = []
            for node in G.nodes():
                if node == start:
                    node_colors.append("tomato")
                elif node == goal:
                    node_colors.append("limegreen")
                elif node in path:
                    node_colors.append("gold")
                else:
                    node_colors.append("lightblue")

            fig, ax = plt.subplots(figsize=(10, 6))
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800, ax=ax)
            nx.draw_networkx_edges(
                G, pos, edgelist=other_edges, edge_color="gray",
                arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.1", ax=ax
            )
            nx.draw_networkx_edges(
                G, pos, edgelist=solution_edges, edge_color="red", width=3,
                arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.1", ax=ax
            )
            nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels=nx.get_edge_attributes(G, "weight"), font_size=7, ax=ax
            )

            ax.set_title(f"{algorithm} Solution Path | Cost: {cost:.2f}")
            ax.axis("off")

            legend_elements = [
                plt.scatter([], [], c="tomato", s=100, label="Start"),
                plt.scatter([], [], c="limegreen", s=100, label="Goal"),
                plt.scatter([], [], c="gold", s=100, label="Path Node"),
                plt.scatter([], [], c="lightblue", s=100, label="Other Node"),
            ]
            ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

            st.pyplot(fig)
