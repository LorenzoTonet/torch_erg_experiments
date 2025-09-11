import networkx as nx
import numpy as np
import random

def gen_two_community_graph(max_nodes_per_community=20, min_nodes_per_community=12, p_intra=0.01, links=1)-> nx.Graph:
    """
    Generate a graph with two communities linked by "links" edges.
    
    Returns:
        G (networkx.Graph): Generated graph with two communities.
        labels (list): Community label for each node.
    """
    
    v = random.randint(min_nodes_per_community, max_nodes_per_community)
    v1 = random.randint(v//2, v-3)
    v2 = v - v1

    g1 = nx.erdos_renyi_graph(v1, p=p_intra)
    g2 = nx.erdos_renyi_graph(v2, p=p_intra)

    G = nx.disjoint_union(g1, g2)

    # Add links between the two communities
    for _ in range(links):
        n1 = random.choice(list(range(v1)))
        n2 = random.choice(list(range(v1, v1+v2)))
        G.add_edge(n1, n2)

    return G
    