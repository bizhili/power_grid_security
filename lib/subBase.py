import numpy as np

def incidence_to_link_connection(H):
    """Convert an incidence matrix H to a link connection matrix A."""
    A = H @ H.T  # Compute shared node connectivity
    np.fill_diagonal(A, 0)  # Remove self-connections
    A = (np.abs(A) > 0).astype(int)  # Convert to binary connection matrix
    return A

def find_connected_components(adj_matrix, linkIndex):
    """
    Finds the number of connected components in an undirected graph and 
    lists the nodes in each component.

    Parameters:
    - adj_matrix (numpy.ndarray): n x n adjacency matrix of the graph

    Returns:
    - num_components (int): Number of connected components
    - components (list of sets): List where each set contains nodes of one component
    """
    n = adj_matrix.shape[0]  # Number of nodes
    visited = set()  # Track visited nodes
    components = []  # List of sets containing nodes of each component

    def dfs(node, component):
        """ Depth-First Search to explore a connected component """
        stack = [node]
        while stack:
            curr = stack.pop()
            if curr not in visited:
                visited.add(curr)
                component.add(linkIndex[curr])
                neighbors = np.where(adj_matrix[curr] == 1)[0]  # Get connected nodes
                stack.extend(neighbors)

    # Traverse all nodes
    for node in range(n):
        if node not in visited:  # Found a new component
            component = set()
            dfs(node, component)
            components.append(component)

    num_components = len(components)
    return num_components, components



def disconnect_substation(A, linkPairs= []):
    Acp= A.copy()
    for pair in linkPairs:
        Acp[pair[1], pair[0]]= 0
        Acp[pair[0], pair[1]]= 0
    return Acp

def create_new_incidence_matrix(A, ALNew):
    newA= A.copy() 
    n= A.shape[1]
    m= A.shape[0]
    splitNodes= dict()
    for i in range(n):
        linkAttached = np.nonzero(np.abs(A)[:, i]>0)[0]
        attachInfo= ALNew[linkAttached, :][:, linkAttached]
        num_components, components= find_connected_components(attachInfo, linkAttached)
        if num_components>1: #node spliting
            newA[:, i]= 0
            component= components.pop(0)
            # print(component)
            splitNodes[i]= [(i, [np.nonzero(np.abs(A)[k, :]>0)[0].tolist() for k in component])]
            for j in component:
                newA[j, i]= -newA[j, :].sum()
            for component in components:
                addCol= np.zeros((m, 1))
                for j in component:
                    addCol[j]= -newA[j, :].sum()
                newA= np.hstack((newA, addCol))
                splitNodes[i].append((newA.shape[1]-1, [np.nonzero(np.abs(A)[k, :]>0)[0].tolist() for k in component]))
    return newA, splitNodes