"""
This module implements the Weisfeiler-Lehman graph kernel (WLGK).
This is an unlabeled implementation of the WLGK, which means that the node labels are ignored.
It also works for directed and undirected graphs.

See paper for more details:
    https://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf
"""
from typing import Dict

import networkx as nx
import numpy as np


def _get_initial_colours(G: nx.Graph) -> Dict[int, int]:
    if not G.is_directed():
        # Set the initial colour of each node to its degree
        return {node: hash(G.degree(node)) for node in G.nodes()}
    # Set the initial colour of each node to a combination of its in-degree and out-degree
    return {
        node: hash(tuple((G.in_degree(node), G.out_degree(node)))) for node in G.nodes()
    }


def _update_colour_counts(
    colour_counts: Dict[int, int], current_node_colours: Dict[int, int]
) -> Dict[int, int]:
    # Update the colour map with the current node colours
    for colour in current_node_colours.values():
        colour_counts[colour] = colour_counts.get(colour, 0) + 1
    return colour_counts


def _new_colour_from_colour_pair(colour_1: int, colour_2: int) -> int:
    # Create a sorted list of the neighbouring colours
    return hash(tuple(sorted([colour_1, colour_2])))


def _new_colour_from_neighbours(
    G: nx.Graph, node: int, node_colours: Dict[int, int]
) -> int:
    # If G is undirected:
    if G.is_directed() is False:
        # Create a sorted list of the neighbouring colours
        colour_list = sorted([node_colours[n] for n in G.neighbors(node)])
        return hash(tuple(colour_list))
    # If G is directed, hash predecessors and successors separately
    colour_list = sorted([node_colours[p] for p in G.predecessors(node)])
    new_colour_predecessors = hash(tuple(colour_list))
    colour_list = sorted([node_colours[s] for s in G.successors(node)])
    new_colour_successors = hash(tuple(colour_list))
    # Combine the hashes
    return hash(tuple((new_colour_predecessors, new_colour_successors)))


def _compute_colour_counts(G: nx.Graph, num_iterations: int) -> Dict[int, int]:
    colour_counts = dict()
    # Set the initial colour of each node to a default colour
    node_colours = _get_initial_colours(G)
    # Update the colour map with the current node colours
    colour_counts = _update_colour_counts(colour_counts, node_colours)
    for _ in range(num_iterations):
        new_node_colours = dict()
        for node in G.nodes():
            # Get the current colour of the node
            old_colour = node_colours[node]
            # Create a new colour containing the colour of each neighbour
            new_colour = _new_colour_from_neighbours(G, node, node_colours)
            # Create an updated colour containing the existing colour and the new colour
            updated_colour_set = _new_colour_from_colour_pair(old_colour, new_colour)
            # Update the node colours
            new_node_colours[node] = updated_colour_set

        # Assign the new node colours
        node_colours = new_node_colours
        # Update the colour counts with the new node colours
        colour_counts = _update_colour_counts(colour_counts, node_colours)

    return colour_counts


def _compute_wlgk_from_colour_counts(
    g1_colour_count: Dict[int, int], g2_colour_count: Dict[int, int]
) -> float:
    # First, create a list of the union of colours
    union_colours = set(g1_colour_count.keys()).union(set(g2_colour_count.keys()))
    # Create a vector for each graph
    g1_colour_vector = np.array([g1_colour_count.get(c, 0) for c in union_colours])
    g2_colour_vector = np.array([g2_colour_count.get(c, 0) for c in union_colours])
    # Normalise the vectors
    norm_g1_colour_vector = g1_colour_vector / np.linalg.norm(g1_colour_vector)
    norm_g2_colour_vector = g2_colour_vector / np.linalg.norm(g2_colour_vector)
    # Compute the dot product
    return np.dot(norm_g1_colour_vector, norm_g2_colour_vector)


def wlgk(G1: nx.Graph, G2: nx.Graph, num_iterations: int = 5) -> float:
    """Compute the Weisfeiler-Lehman graph kernel (WLGK) between two graphs.

    Args:
        G1 (nx.Graph): The first graph.
        G2 (nx.Graph): The second graph.
        num_iterations (int, optional): The number of iterations to run the WLGK for. Defaults to 5.

    Returns:
        float: The WLGK between the two graphs.
    """

    # Check that the graphs aren't empty
    if len(G1.nodes()) == 0 or len(G2.nodes()) == 0:
        # Raise a warning
        print("Warning: One or both graphs have no nodes. Returning 1.0.")
        return 1.0

    g1_colour_count = _compute_colour_counts(G1, num_iterations)
    g2_colour_count = _compute_colour_counts(G2, num_iterations)

    return _compute_wlgk_from_colour_counts(g1_colour_count, g2_colour_count)
