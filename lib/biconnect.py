import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootY] = rootX

def optimal_threshold2(m, leafLinks, cosSim):
    maskMat= np.ones_like(cosSim)- np.eye(m)
    edge2Bi= {i:i for i in range(m) if i not in leafLinks}
    bi2Edges= {i:[i] for i in range(m) if i not in leafLinks}
    valueRecord= []
    while len(bi2Edges.keys())>1:
        idxMax= np.argmax(cosSim*maskMat)
        i= idxMax//m
        j= idxMax%m
        valueRecord.append(cosSim[i, j])
        bii= edge2Bi[i]
        bij= edge2Bi[j]
        bi2Edges[bii]= bi2Edges[bii]+bi2Edges[bij]
        for l in bi2Edges[bij]:
            edge2Bi[l]= bii
            for k in bi2Edges[bii]:
                maskMat[k, l]= 0
                maskMat[l, k]= 0
        bi2Edges.pop(bij)
    valueRecord= np.array(valueRecord)
    # print("record:", valueRecord)
    lenOfnullGap= valueRecord[:-1]/valueRecord[1:]
    idx= np.argmax(lenOfnullGap)
    if lenOfnullGap[idx]>10:
        threhold2= (valueRecord[idx]+valueRecord[idx+1])/2
    else:
        threhold2= 1e-14
    return threhold2


def combine_sets(sets):
    # Mapping each element to its corresponding set index
    element_to_index = {}
    for index, s in enumerate(sets):
        for element in s:
            if element in element_to_index:
                element_to_index[element].add(index)
            else:
                element_to_index[element] = {index}
    
    # Initialize union-find structure
    uf = UnionFind(len(sets))
    
    # Union sets that share common elements
    for indices in element_to_index.values():
        indices = list(indices)
        for i in range(1, len(indices)):
            uf.union(indices[0], indices[i])
    
    # Combine the sets
    new_sets = {}
    for set_index, s in enumerate(sets):
        root = uf.find(set_index)
        if root in new_sets:
            new_sets[root].update(s)
        else:
            new_sets[root] = set(s)

    return list(new_sets.values())

def gen_biconnectedlist(newDataNp):
    U, S, _ = np.linalg.svd(newDataNp.T, full_matrices=True) 
    m=newDataNp.shape[1]
    #get rank
    SGap= S[1:]/S[:-1]
    rank= np.argmin(SGap)+1
    #get threshold1 
    nullSapce= U[:, rank:]
    cosSim= np.abs(cosine_similarity(nullSapce, nullSapce))
    lenOfnull= np.sqrt(np.square(nullSapce).sum(axis= 1))
    lenOfnull= np.sort(lenOfnull)
    lenOfnullGap= lenOfnull[1:]/lenOfnull[:-1]
    maxIdx= np.argmax(lenOfnullGap)
    print("LEN:", len(lenOfnull))
    if lenOfnullGap[maxIdx]>2:
        threshold1= (lenOfnull[maxIdx]+lenOfnull[maxIdx+1])/2
    else:
        threshold1= 1e-15
    # get leaf
    leafEntry= 1*((np.sqrt(np.square(nullSapce).sum(axis= 1)))<threshold1)
    leafLinks= np.nonzero(leafEntry)[0].tolist()
    cosSim[:, leafLinks]= 0
    cosSim[leafLinks, :]= 0
    #get threshold2
    threshold2= optimal_threshold2(m, leafLinks, cosSim)
    # print(threshold2)
    # find biconnected components and leaf
    biconnectedConps = (cosSim> threshold2)

    biconnectedList = [[it] for it in leafLinks]
    edgeList= [i for i in range(m) if i not in leafLinks]
    while len(edgeList):
        i= edgeList[0]
        edgeItem= np.nonzero(biconnectedConps[i, :])[0].tolist()
        biconnectedList.append(edgeItem)
        edgeList.pop(edgeList.index(i))
    biconnectedList= combine_sets(biconnectedList)
    biconnectedList= [list(i) for i in biconnectedList]
    return biconnectedList, cosSim


