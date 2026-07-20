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

def check_spanning_tree(H, spaningTreeList, rankH):
    if np.linalg.matrix_rank(H[spaningTreeList, :]) ==rankH:
        print("Correct spaning tree")
    else:
        print("Incorrect spaning tree")

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

def check_tree_links(A, rankH, treeLinks):
    U, _, _= np.linalg.svd(A)
    nullA= U[:, rankH:]
    realTree= set(np.where(np.linalg.norm(nullA, axis= 1)<1e-10)[0].tolist())
    interSect= realTree.intersection(treeLinks)
    print("Recall tree:", len(interSect)/len(realTree))
    print("Precision tree:", len(interSect)/(len(treeLinks)))


def find_components(sets):
    from collections import defaultdict, deque
    
    # Build the adjacency list based on intersections
    n = len(sets)
    adj = defaultdict(set)
    
    for i in range(n):
        for j in range(i+1, n):
            if sets[i] & sets[j]:  # intersection is non-empty
                adj[i].add(j)
                adj[j].add(i)
    
    visited = set()
    components = []
    
    # BFS to find connected components
    for i in range(n):
        if i not in visited:
            component = set()
            queue = deque([i])
            visited.add(i)
            while queue:
                node = queue.popleft()
                component.add(node)
                for neighbor in adj[node]:
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
    predParas, cSpace= parasLearn.paras_learning(biComponents, wholeRight, newDataNp)

    # print(cycleList)
    return predParas, cSpace, wholeRight