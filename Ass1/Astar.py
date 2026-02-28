
#The script loads a driving network from OpenStreetMap (using osmnx), sets each edge’s travel time as its 
#weight (length / maxspeed), runs a Dijkstra shortest-path search (using heapq) between two randomly chosen nodes, 
#styles edges/nodes for visualization while the algorithm runs, reconstructs the found path, increments an 
#edge usage counter (dijkstra_uses) for the path, and finally draws a graph visualization showing visited/active/path 
#edges. The graph object G is a global osmnx MultiDiGraph.

import osmnx as ox
import random
import heapq
import math
import matplotlib.pyplot as plt

## TODO complete this file with implementation of A* starting from Dijkstra's implementation in Dijkstra.py.


def manhattan_distance(point1, point2):
    """Compute manhattan distance between two points.
    
    Formula:
        h(n) = |x_1 - x_2| + |y_1 - y_2|

    Args:
        point1: A tuple representing the coordinates of the first point (x1, y1).
        point2: A tuple representing the coordinates of the second point (x2, y2).
    Returns:
        The euclidean distance between the two points.
    """
    x1, y1 = point1
    x2, y2 = point2
    return abs(x1 - x2) + abs(y1 - y2)


def euclidean_distance(point1, point2):
    """Compute euclidean distance between two points.

    Formula:
        d = sqrt((x_1 - x_2)^2 + (y_1 - y_2)^2)

    Args:
        point1: A tuple representing the coordinates of the first point (x1, y1).
        point2: A tuple representing the coordinates of the second point (x2, y2).
    Returns:
        The euclidean distance between the two points.
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt(((x1 - x2) ** 2 + (y1 - y2) ** 2))


def haversine_distance(point1, point2):
    """Compute haversine distance between two points.

    Formula:
        a = sin^2((lat2 - lat1) / 2) + cos(lat1) * cos(lat2) * sin^2((lon2 - lon1) / 2)
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        d = R * c

    Args:
        point1: A tuple representing the coordinates of the first point (lat1, lon1).
        point2: A tuple representing the coordinates of the second point (lat2, lon2).
    Returns:
        The haversine distance between the two points in kilometers.
    """
    lat1, lon1 = point1
    lat2, lon2 = point2
    R = 6371

    phi1 = math.radians(lat1)
    #phi1 = lat1
    phi2 = math.radians(lat2)
    #phi2 = lat2
    delta_phi = math.radians(lat1 - lat2)
    delta_lambda = math.radians(lon1 - lon2)

    a = (math.sin(delta_phi / 2) ** 2) + math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c


def style_unvisited_edge(edge):        
    G.edges[edge]["color"] = "gray"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 0.2

def style_visited_edge(edge):
    G.edges[edge]["color"] = "green"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def style_active_edge(edge):
    G.edges[edge]["color"] = "red"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def style_path_edge(edge):
    G.edges[edge]["color"] = "white"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 5

def plot_graph():
    ox.plot_graph(
        G,
        node_size =  [ G.nodes[node]["size"] for node in G.nodes ],
        edge_color = [ G.edges[edge]["color"] for edge in G.edges ],
        edge_alpha = [ G.edges[edge]["alpha"] for edge in G.edges ],
        edge_linewidth = [ G.edges[edge]["linewidth"] for edge in G.edges ],
        node_color = "white",
        bgcolor = "black"
    )
def save_graph(path, title=None):
    fig, ax = ox.plot_graph(
        G,
        node_size =  [ G.nodes[node]["size"] for node in G.nodes ],
        edge_color = [ G.edges[edge]["color"] for edge in G.edges ],
        edge_alpha = [ G.edges[edge]["alpha"] for edge in G.edges ],
        edge_linewidth = [ G.edges[edge]["linewidth"] for edge in G.edges ],
        node_color = "white",
        bgcolor = "black",
        save=False, 
        show=False
    )
    if title:
        ax.set_title(title, color="white", fontsize=14)
    fig.savefig(f"{path}.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

def a_star(orig, dest, heuristic, scaling, plot=False):
    """A* algorithm implementation.
    Args:
        orig: Starting node.
        dest: Destination node.
        heuristic: G function that takes two points and returns an estimate of the cost to reach dest from the point.
    Returns:
        Number of iterations taken to find the path, None if no path was found.
    """

    #initialize nodes and edges
    for node in G.nodes:
        G.nodes[node]["visited"] = False
        G.nodes[node]["distance"] = float("inf")
        G.nodes[node]["previous"] = None
        G.nodes[node]["size"] = 0
    for edge in G.edges:
        style_unvisited_edge(edge) #as dijkstra
    G.nodes[orig]["distance"] = 0
    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50
    pq = [(0, orig)]
    step = 0
    while pq:
        _, node = heapq.heappop(pq)
        if node == dest:
            #print("Iterations:", step)
            #plot_graph()
            return step
        
        if G.nodes[node]["visited"]:
            continue

        G.nodes[node]["visited"] = True
        for edge in G.out_edges(node):
            style_visited_edge((edge[0], edge[1], 0))
            neighbor = edge[1]
            weight = G.edges[(edge[0], edge[1], 0)]["weight"]
            if G.nodes[neighbor]["distance"] > G.nodes[node]["distance"] + weight:
                G.nodes[neighbor]["distance"] = G.nodes[node]["distance"] + weight
                G.nodes[neighbor]["previous"] = node
                ###### NAIVE IMPLEMENTATION (WITHOUT PROPER HEURISTIC WEIGHT) ######
                #heuristic_cost = heuristic((G.nodes[neighbor]['y'], G.nodes[neighbor]['x']), (G.nodes[dest]['y'], G.nodes[dest]['x']))
                #heapq.heappush(pq, (G.nodes[neighbor]["distance"] + heuristic_cost, neighbor))
                heuristic_cost = heuristic(
                        (G.nodes[neighbor]['y'], G.nodes[neighbor]['x']), 
                        (G.nodes[dest]['y'], G.nodes[dest]['x'])
                    )*scaling #SCALE HEURISTIC TO MAX SPEED TO AVOID OVERWEIGHTING
                heapq.heappush(pq, (G.nodes[neighbor]["distance"] + heuristic_cost, neighbor))
                for edge2 in G.out_edges(neighbor):
                    style_active_edge((edge2[0], edge2[1], 0))
        step += 1
    print("--> No path found")

def dijkstra(orig, dest, plot=False):
    for node in G.nodes:
        G.nodes[node]["visited"] = False
        G.nodes[node]["distance"] = float("inf")
        G.nodes[node]["previous"] = None
        G.nodes[node]["size"] = 0
    for edge in G.edges:
        style_unvisited_edge(edge)
    G.nodes[orig]["distance"] = 0
    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50
    pq = [(0, orig)]
    step = 0
    while pq:
        _, node = heapq.heappop(pq)
        if node == dest:
            #print("Iterations:", step)
            #plot_graph()
            return step
        if G.nodes[node]["visited"]: continue
        G.nodes[node]["visited"] = True
        for edge in G.out_edges(node):
            style_visited_edge((edge[0], edge[1], 0))
            neighbor = edge[1]
            weight = G.edges[(edge[0], edge[1], 0)]["weight"]
            if G.nodes[neighbor]["distance"] > G.nodes[node]["distance"] + weight:
                G.nodes[neighbor]["distance"] = G.nodes[node]["distance"] + weight
                G.nodes[neighbor]["previous"] = node
                heapq.heappush(pq, (G.nodes[neighbor]["distance"], neighbor))
                for edge2 in G.out_edges(neighbor):
                    style_active_edge((edge2[0], edge2[1], 0))
        step += 1
    print("--> No path found")

def reconstruct_path(orig, dest, plot=False, algorithm=None):
    if G.nodes[dest]["previous"] is None and dest != orig:
        print("No path found")
        return None
    for edge in G.edges:
        style_unvisited_edge(edge)
    dist = 0
    speeds = []
    curr = dest
    while curr != orig:
        prev = G.nodes[curr]["previous"]
        dist += G.edges[(prev, curr, 0)]["length"]
        speeds.append(G.edges[(prev, curr, 0)]["maxspeed"])
        style_path_edge((prev, curr, 0))
        if algorithm:
            G.edges[(prev, curr, 0)][f"{algorithm}_uses"] = G.edges[(prev, curr, 0)].get(f"{algorithm}_uses", 0) + 1
        curr = prev
    dist /= 1000

def graph_init():
    for edge in G.edges:
    # Cleaning the "maxspeed" attribute, some values are lists, some are strings, some are None
        maxspeed = 40
        if "maxspeed" in G.edges[edge]:
            maxspeed = G.edges[edge]["maxspeed"]
            if type(maxspeed) == list:
            #speeds = [ int(speed) for speed in maxspeed ]
                speeds = [int(speed) if speed != "walk" else 1 for speed in maxspeed]
                maxspeed = min(speeds)
            elif type(maxspeed) == str:
                if maxspeed == "walk": 
                    maxspeed = 1
                else:
                    maxspeed = maxspeed.strip(" mph")
                    maxspeed = int(maxspeed)
        G.edges[edge]["maxspeed"] = maxspeed
    # Adding the "weight" attribute (time = distance / speed)
        G.edges[edge]["weight"] = G.edges[edge]["length"] / maxspeed


    for edge in G.edges:
        G.edges[edge]["astar_uses"] = 0

def compare_graphs_overlay(place_name):
    """Shows in red nodes/edges removed from original graph"""
    

    G_original = ox.graph_from_place(place_name, network_type="drive")
    G_truncated = ox.truncate.largest_component(G_original, strongly=True)
    
    removed_nodes = set(G_original.nodes) - set(G_truncated.nodes)
    removed_edges = set(G_original.edges) - set(G_truncated.edges)
    
    print(f"Original:  {len(G_original.nodes)} nodes, {len(G_original.edges)} edges")
    print(f"Truncated:   {len(G_truncated.nodes)} nodes, {len(G_truncated.edges)} edges")
    print(f"Removed:    {len(removed_nodes)} nodes, {len(removed_edges)} edges")
    
    # Plot overlay
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Original graph
    ox.plot_graph(G_truncated, ax=ax, node_size=0, edge_color="gray",
                  edge_linewidth=0.5, bgcolor="black", show=False)
    
    # Differenees (edge/node removed)
    if removed_edges:
        ox.plot_graph(G_original, ax=ax, node_size=0, 
                      edge_color=["red" if e in removed_edges else "none" for e in G_original.edges],
                      edge_linewidth=2, bgcolor="black", show=False)
    
    if removed_nodes:
        node_colors = ["red" if n in removed_nodes else "none" for n in G_original.nodes]
        node_sizes = [50 if n in removed_nodes else 0 for n in G_original.nodes]
        ox.plot_graph(G_original, ax=ax, node_color=node_colors, node_size=node_sizes,
                      edge_color="none", bgcolor="black", show=False)
    
    ax.set_title(f"Gray: connected | Red: removed ({len(removed_nodes)} nodes, {len(removed_edges)} edges)", 
                 color="white")
    fig.patch.set_facecolor("black")
    plt.savefig(f"{place_name}_overlay.png", bbox_inches="tight", facecolor=fig.get_facecolor())

#compare_graphs_overlay("Aosta, Aosta, Italy")


if __name__ == "__main__":

    places = ["Turin, Piedmont, Italy", "Aosta, Aosta, Italy"]
    for place_name in places:
        astar_iterations = []

        G = ox.graph_from_place(place_name, network_type="drive")

        ## Keep only the largest strongly connected component (the one where you can reach any node from any other)
        G = ox.truncate.largest_component(G, strongly=True)
        graph_init()
        MAX_SPEED = max(G.edges[edge]["maxspeed"] for edge in G.edges)

        ## Fixes overweight of heuristic (necessary in order to make it admissible)
        # 
        avg_lat = math.radians(sum(G.nodes[n]['y'] for n in G.nodes) / len(G.nodes))
        lon_scale = math.cos(avg_lat)

        heuristics = [
            (euclidean_distance, 111_000*lon_scale/MAX_SPEED), 
            (manhattan_distance, 111_000*lon_scale/MAX_SPEED), 
            (haversine_distance, 1_000/MAX_SPEED)]

        start_end_pairs = []
        for i in range(10):
            start = random.choice(list(G.nodes))
            end = random.choice(list(G.nodes))
            start_end_pairs.append((start, end))
        
        print(f"Number of edges in {place_name}:", len(G.edges))
        print(f"Number of nodes in {place_name}:", len(G.nodes))

        for heuristic, scaling in heuristics:
            graph_init()
            astar_iterations = []
            for start, end in start_end_pairs:

                #start = random.choice(list(G.nodes))
                #end = random.choice(list(G.nodes))

                #print(f"Running Astar from {start} to {end} in {place_name}...")
                iterations = a_star(start, end, heuristic, scaling=scaling)
                astar_iterations.append(iterations)
                #print("Iterations:", iterations)
                reconstruct_path(start, end, algorithm="astar", plot=False)

            print(f"[{heuristic.__name__}] Average iterations for A* in {place_name}: {sum(astar_iterations) / len(astar_iterations)}")

            ## graph visualization
            for edge in G.edges:
                uses = G.edges[edge].get("astar_uses", 0)
                if uses > 0:
                    G.edges[edge]["color"] = "red"
                    G.edges[edge]["alpha"] = 1
                    G.edges[edge]["linewidth"] = 1 + uses
                else:
                    style_unvisited_edge(edge)
        
            save_graph(heuristic.__name__ + "_" + place_name.replace(", ", "_").replace(" ", "_"), title=f"A* with {heuristic.__name__} heuristic in {place_name}")

