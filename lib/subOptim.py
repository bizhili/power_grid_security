import numpy as np

def find_paths(adj_matrix, start, steps, path, visited, all_paths):
    if steps == 0:  # Base case: required number of steps reached
        all_paths.append(path[:])
        return  
    for neighbor, is_edge in enumerate(adj_matrix[start]):
        if is_edge and neighbor not in visited:  # Check if an edge exists and node is not visited
            visited.add(neighbor)
            path.append(neighbor)
            find_paths(adj_matrix, neighbor, steps - 1, path, visited, all_paths)
            path.pop()  # Backtrack
            visited.remove(neighbor)

def last_step(initial_node_neighbors, all_paths):
    left_paths= []
    for path in all_paths:
        if path[-1] in initial_node_neighbors:
            left_paths.append(path)
    return left_paths


def get_all_paths(adj_matrix, start, steps, singledir= []):
    all_paths = []
    adj_matrix_cp= adj_matrix.copy()
    adj_matrix_cp[singledir, :]= 0 # single direction, set end nodes to 0
    find_paths(adj_matrix_cp, start, steps, [start], {start}, all_paths)
    initial_node_neighbors = np.nonzero(adj_matrix_cp[start, :])[0]
    left_paths= last_step(initial_node_neighbors, all_paths) # get all paths that end in the initial node
    return left_paths