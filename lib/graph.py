import numpy as np
import networkx as nx

def from_A_to_G(A):
    # Create an empty directed graph
    G = nx.DiGraph()

    # Get the number of edges and nodes
    num_edges, num_nodes = A.shape

    # Add nodes to the graph
    G.add_nodes_from(range(num_nodes))

    # node pair to link id
    node_pair2id= dict()

    # Add edges based on the topology matrix A
    for i in range(num_edges):
        source = np.where(A[i] == 1)[0]
        target = np.where(A[i] == -1)[0]
        if source.size > 0 and target.size > 0:
            G.add_edge(source[0], target[0], label= i) #here label is the label
            node_pair2id[(source[0], target[0])]= i
            node_pair2id[(target[0], source[0])]= i
    return G.to_undirected(), node_pair2id 