from scipy.linalg import qr
import numpy as np
import lib.parasLearn as parasLearn


def select_rows_rrqr(H_e, rank=13):
    _, _, pivot = qr(H_e, pivoting=True)
    selected_rows = pivot[:rank]
    return selected_rows.tolist()

def linksDependent(covM):
    Val, Vt = np.linalg.eig(covM)
    Val= np.abs(np.real(Val))
    minIdx= np.argmin(Val)
    nullSpace= Vt[:, minIdx:minIdx+1].T
    nullMagni= np.linalg.norm(nullSpace, axis= 0)
    return nullMagni

def get_spanning_tree(dataZ, rankH, m):
    spaningTreeList= select_rows_rrqr(dataZ, rank= rankH)
    leftLinksSet= set([i for i in range(m)])-set(spaningTreeList)
    return spaningTreeList, leftLinksSet

def check_spanning_tree(A, spaningTreeList, rankH):
    """Return whether the selected links form a spanning tree."""
    links = list(spaningTreeList)
    valid_ids = (
        len(links) == len(set(links))
        and all(isinstance(i, (int, np.integer)) and 0 <= i < A.shape[0] for i in links)
    )
    correct = (
        valid_ids
        and np.linalg.matrix_rank(A) == rankH
        and len(links) == rankH
        and np.linalg.matrix_rank(A[links, :]) == rankH
    )
    print("spaningTreeList correct:", correct)
    return correct

def get_cycle_and_tree(spaningTreeList, leftLinksSet, wholeRight, m):
    cycleList= []
    leftLinksList= list(leftLinksSet)
    treeLinks= set([i for i in range(m)])# links in a tree structure, not belong to any cycle
    Sss= []
    logSss= []
    for i, link in  enumerate(leftLinksList):
        tmpLinks= spaningTreeList+[link]
        covM= wholeRight[tmpLinks, :][:, tmpLinks]
        nullMagni= linksDependent(covM)
        rankedIdx= np.argsort(nullMagni)
        convRankedIdx= [tmpLinks[i] for i in reversed(rankedIdx)]
        Ss= np.zeros(len(convRankedIdx))# used to be -1
        for j in range(1, len(convRankedIdx)+1): # used to be +0
            squareRight= wholeRight[:, convRankedIdx[:j]][convRankedIdx[:j], :]# right covariance matrix
            singularVs = np.sqrt(np.abs(np.linalg.eigvals(squareRight)))# transform eigenvalue to sigular value 
            Ss[j-1]= np.min(singularVs)
        Sss.append(Ss)
        logSs= np.log(Ss+1e-9)
        logSss.append(logSs)
        critia= (logSs[:-2]+logSs[2:]-2*logSs[1:-1])/(Ss[1:-1]+1e-9)
        maxId= np.argmax(critia)+2
        cycleTmp= convRankedIdx[:maxId]
        treeLinks= treeLinks-set(cycleTmp)
        cycleList.append(set(cycleTmp))
    return cycleList, treeLinks

def get_bridge_links(A):
    """Return links that belong to no cycle (graph bridges)."""
    rankA = np.linalg.matrix_rank(A)
    return {
        i for i in range(A.shape[0])
        if np.linalg.matrix_rank(np.delete(A, i, axis=0)) < rankA
    }


def check_tree_links(A, rankH, treeLinks):
    """Return whether treeLinks contains exactly all graph bridges."""
    expected = get_bridge_links(A)
    correct = np.linalg.matrix_rank(A) == rankH and set(treeLinks) == expected
    print("treeLinks correct:", correct)
    if not correct:
        print("missing:", sorted(expected - set(treeLinks)))
        print("extra:", sorted(set(treeLinks) - expected))
    return correct


def check_cycle_list(A, cycleList):
    """Check that every inferred link set is exactly one graph cycle."""
    incorrect = []
    for cycle_id, cycle in enumerate(cycleList):
        links = list(cycle)
        valid_ids = (
            len(links) == len(set(links))
            and all(isinstance(i, (int, np.integer)) and 0 <= i < A.shape[0] for i in links)
        )
        correct = valid_ids and len(links) > 1
        if correct:
            cycle_rank = np.linalg.matrix_rank(A[links, :])
            correct = (
                cycle_rank == len(links) - 1
                and all(
                    np.linalg.matrix_rank(A[links[:i] + links[i + 1:], :]) == cycle_rank
                    for i in range(len(links))
                )
            )
        if not correct:
            incorrect.append((cycle_id, sorted(links)))

    print("cycleList correct:", not incorrect)
    for cycle_id, links in incorrect:
        print(f"incorrect cycleList[{cycle_id}] links:", links)
    return not incorrect


def find_components(sets):
    from collections import defaultdict, deque
    
    # Build the adjacency list based on intersections
    nCyc = len(sets)
    adj = defaultdict(set)
    
    for i in range(nCyc):
        for j in range(i+1, nCyc):
            if sets[i] & sets[j]:  # intersection is non-empty
                adj[i].add(j)
                adj[j].add(i)
    
    visited = set()
    components = []
    
    # BFS to find connected components
    for i in range(nCyc):
        if i not in visited:
            component = set()
            queue = deque([i])
            visited.add(i)
            while queue:
                cycle = queue.popleft()
                component.add(cycle)
                for neighbor in adj[cycle]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
    # Return the actual sets grouped by components
    grouped_components = [[list(sets[i]) for i in component] for component in components]
    return grouped_components


def get_cycle_space(newDataNp, rankH, m, cycleList= None):
    wholeRight= newDataNp.T.dot(newDataNp)
    if cycleList== None:
        spaningTreeList, leftLinksSet = get_spanning_tree(newDataNp, rankH, m)
        cycleList, _ = get_cycle_and_tree(spaningTreeList, leftLinksSet, wholeRight, m)
    biComponents= find_components(cycleList.copy())
    predParas, cSpace= parasLearn.paras_learning(biComponents, wholeRight)

    # print(cycleList)
    return predParas, cSpace, wholeRight
