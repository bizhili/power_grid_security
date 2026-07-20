import numpy as np


def generate_random_power_flow(H, dataNum):
    newData= []
    for i in range(dataNum):
        data= H.dot(np.random.randn(H.shape[1], 1))
        newData.append(data)
    newDataNp= np.stack(newData).squeeze()
    return newDataNp

def add_noise_to_data(newDataNp, stdOfNoise):
    np.random.seed(1)
    newDataNpT= newDataNp.copy()+np.random.randn(newDataNp.shape[0], newDataNp.shape[1])*stdOfNoise
    return newDataNpT

def node_power_from_branch_ends(A, pf_from, pf_to):
    """
    Compute nodal net power from branch-end flows.
    A: (m, n) incidence with +1 at source (from) node, -1 at target (to) node
    pf_from: (m,) or (m,k) power at from ends (positive = leaves the bus into the line)
    pf_to:   (m,) or (m,k) power at to ends   (positive = leaves the bus into the line)
             (can be real for P, or complex for S=P+jQ; shapes must match)

    Returns:
      p_node: (n,) or (n,k) net power into each node from connected lines
              (positive means net line-to-bus injection; negative means bus-to-line export)
    """
    A_plus  = (A ==  1).astype(float)  # selects from-end bus for each branch
    A_minus = (A == -1).astype(float)  # selects to-end bus for each branch

    # Sum “leaving the bus” contributions per node, then flip sign to get “into the bus”
    node_out = A_plus.T @ pf_from + A_minus.T @ pf_to   # total leaving each node into lines
    p_node   = -node_out                                # net into node from lines
    return p_node