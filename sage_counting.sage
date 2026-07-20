from sage.graphs.connectivity import spqr_tree
import re
G = Graph()
G.add_edges([
    [0, 1], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [3, 4], [3, 6], [3, 8], [4, 5], [5, 10], [5, 11], [5, 12], [6, 8], [8, 9], [8, 13], [9, 10], [11, 12], [12, 13],
])
Tree = spqr_tree(G)
G.show()
labels = Tree.vertices()
Slist= []
nR= 0
for node in Tree:
    label= node[0]
    feature= str(node[1])
    match = re.search(r"\b\d+(\.\d+)?\b", feature)
    if label== 'S':
        Slist.append(int(match.group()))
    elif label== 'R':
        nR+=1
print(Slist)
print(nR)
product = 1
for sCycLen in Slist:
    product *= factorial(sCycLen - 1)
result = max(1, (2 ** (nR - 1)) * product)
print("number of 2-isomorphic:", result)
autList= G.automorphism_group().list()
result2= len(autList)
print("number of auto-isomorphic:", result2)
n= len(G)
#result3= factorial(n)/len(autList)
print(f"number of 1-isomorphic:{n}!/{len(autList)}")