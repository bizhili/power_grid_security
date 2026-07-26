import numpy as np
from collections import deque
from typing import List, Tuple


def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """Return the signal-to-noise power ratio in decibels."""
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = np.mean(np.abs(noise) ** 2)
    return np.inf if noise_power == 0 else 10 * np.log10(signal_power / noise_power)


def bfs_spanning_tree_from_incidence(
    A: np.ndarray,
    root: int = 0,
    allow_forest: bool = False,
) -> Tuple[List[int], np.ndarray]:
    """
    Build a spanning tree (or forest) from an incidence matrix using BFS.

    Parameters
    ----------
    A : np.ndarray
        Incidence matrix of shape (n_nodes, n_edges). Each column should have
        exactly two non-zero entries for a simple edge (±1 or 1/1). Columns
        with 0 or 1 non-zero (self-loop) are ignored.
    root : int, default=0
        Start node for BFS.
    allow_forest : bool, default=False
        If True, continues BFS from remaining unvisited nodes to produce a forest.
        If False, raises if the graph is disconnected.

    Returns
    -------
    tree_edge_indices : List[int]
        List of edge (column) indices that form the spanning tree (or forest).
    parent : np.ndarray
        Parent array of shape (n_nodes,), with -1 for the roots and parent[u]=v otherwise.

    Raises
    ------
    ValueError
        If the graph is disconnected and allow_forest is False,
        or if root is out of range.
    """
    A = np.asarray(A)
    n_nodes, n_edges = A.shape
    if not (0 <= root < n_nodes):
        raise ValueError("root out of range")

    # Build undirected neighbor lists from incidence
    neighbors = [[] for _ in range(n_nodes)]  # neighbors[u] = list of (v, edge_idx)
    for e in range(n_edges):
        nodes = np.flatnonzero(A[:, e])
        if nodes.size == 2:
            u, v = int(nodes[0]), int(nodes[1])
            if u != v:  # ignore self-loops
                neighbors[u].append((v, e))
                neighbors[v].append((u, e))
        # else: ignore malformed columns (0 or >2 incident nodes)

    visited = np.zeros(n_nodes, dtype=bool)
    parent = -np.ones(n_nodes, dtype=int)
    tree_edges: List[int] = []

    def bfs(start: int):
        if visited[start]:
            return
        visited[start] = True
        q = deque([start])
        while q:
            u = q.popleft()
            for v, e in neighbors[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    tree_edges.append(e)
                    q.append(v)

    bfs(root)

    if not visited.all():
        if allow_forest:
            # Continue BFS from remaining components
            for s in range(n_nodes):
                if not visited[s]:
                    bfs(s)
        else:
            raise ValueError("Graph is disconnected; set allow_forest=True to return a forest.")

    return tree_edges, parent
