import numpy as np
import networkx as nx

def from_A_to_G(A, P=None):
    # Create an empty directed graph
    G = nx.Graph()

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
            Sint= int(source[0])
            Tint= int(target[0])
            if P is None:
                G.add_edge(Sint, Tint, label= int(i)) #here label is the label
            else:
                G.add_edge(Sint, Tint, label= int(i), weight= P[i]) #here label is the label
            node_pair2id[(Sint, Tint)]= i
            node_pair2id[(Tint, Sint)]= i
    return G.to_undirected(), node_pair2id 