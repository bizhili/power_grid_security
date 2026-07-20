import lib.biconnect as biconnect
import numpy as np

def num_circle(checkData, checkIDs):
    if len(checkIDs)<3:
        return 0, None
    tempData= np.copy(checkData[:, list(checkIDs)])
    U, Sigma, _= np.linalg.svd(tempData.T, full_matrices=True)
    # print("Sigma:", Sigma)
    SigmaGap= Sigma[1:]/Sigma[:-1]
    minIdx= np.argmin(SigmaGap)
    rank= minIdx+1
    if SigmaGap[minIdx]>1e-1: # gap between, related to noise level
        # print(Sigma)
        rank= Sigma.shape[0]
    numOfCycles= len(checkIDs)-rank
    treeLinks= []
    # if number of cycle is 1, detect tree links
    if numOfCycles== 1:
        tmpNullSpace= U[:, rank:]
        lenOfnull= np.sqrt(np.square(tmpNullSpace).sum(axis= 1))
        lenOfnull= np.sort(lenOfnull)
        lenOfnullGap= lenOfnull[1:]/lenOfnull[:-1]
        maxIdx= np.argmax(lenOfnullGap)
        if lenOfnullGap[maxIdx]>10:
            threshold1= (lenOfnull[maxIdx]+lenOfnull[maxIdx+1])/2
        else:
            # print((lenOfnull[maxIdx]+lenOfnull[maxIdx+1])/2)
            threshold1= 1e-15
        # get leaf
        leafEntry= 1*((np.sqrt(np.square(tmpNullSpace).sum(axis= 1)))<threshold1)
        treeLinks= np.nonzero(leafEntry)[0].tolist()
        treeLinks= [checkIDs[i] for i in treeLinks]
        # print("Sigma:", Sigma)
        # print(checkIDs)
        # print(lenOfnull)
        # print(treeLinks)
    return numOfCycles, treeLinks

def extract_circles_full_edge(biconnectedConp, testData, cosSim):
    edgeList= biconnectedConp.copy()
    anyCircles= []
    countRun= 0
    while len(edgeList):
        # print(len(edgeList))
        i= edgeList.pop(-1)
        curCos= 1-cosSim[i, :].copy()
        rankedEdge= np.argsort(curCos).tolist() #rank neighbor links by cosine similarity
        rankedEdge.pop(rankedEdge.index(i)) # remove it self
        curEdges= [i] # detect cache
        for jt in rankedEdge:
            if jt in biconnectedConp: # add this neighbor to detect cache
                curEdges.append(jt)
            if len(curEdges)>2: # detect if number of links in cache is more than 2
                countRun+=1 # regard as on operation
                num_circles, edgeSetTmp= num_circle(testData, curEdges) # return number of cycles composing in these links, may include tree links
                if num_circles== 1: #contain one cycle
                    tmpCircle= set(curEdges)-set(edgeSetTmp)
                    if tmpCircle not in anyCircles:
                            anyCircles.append(tmpCircle.copy())
                            for jt in tmpCircle:
                                if jt in edgeList:
                                    edgeList.pop(edgeList.index(jt))
                    if i in edgeSetTmp:
                        curEdges.pop(-1)
                    else:
                        break
    return anyCircles, countRun

def get_left_circle_edges(combinedCircles, testData):
    combinedCirclescp= [list(i) for i in combinedCircles]
    circleAfterSum= 1
    while circleAfterSum:
        circleAfterSum= 0
        for it in combinedCirclescp:
            # print(combinedCirclescp)
            if len(it)==1:
                continue
            circleBefore, _= num_circle(testData, it)
            sumCircle= []
            for jt in combinedCirclescp:
                sumCircle+= jt
            circleWholeBefore, _= num_circle(testData, sumCircle)
            delLink= it.pop(0)
            circleAfter, _= num_circle(testData, it)
            circleAfterSum+= circleAfter
            sumCircle= []
            for jt in combinedCirclescp:
                sumCircle+= jt
            circleWholeAfter, _= num_circle(testData, sumCircle)
            if circleWholeBefore-circleWholeAfter!=circleBefore-circleAfter:
                it.append(delLink)
    sumCircle= []
    for jt in combinedCirclescp:
        sumCircle+= jt
    return sumCircle


def get_biconnected_components_dict(newDataNp, cosSim, biconnectedList):
    biconnected_components_dict= dict()
    for biconnectedConp in biconnectedList:
        if len(biconnectedConp)== 1:
            continue
        anyCirclesSum =[] # store all dicted cycles
        linksFoucs= biconnectedConp.copy() # left links not in any cycle
        countRunSum= 0 # number of operations
        lastFoucs= linksFoucs.copy()
        while(True):
            anyCircles, countRun= extract_circles_full_edge(linksFoucs, newDataNp, cosSim)
            anyCirclesSum+= anyCircles
            countRunSum+= countRun
            combinedCircles= biconnect.combine_sets(anyCirclesSum)
            if len(combinedCircles)== 1:
                break
            if len(combinedCircles)== 0:
                return dict()
            linksFoucs= get_left_circle_edges(combinedCircles, newDataNp)
            if lastFoucs==linksFoucs:
                return dict()
            lastFoucs= linksFoucs.copy()
            # print(len(linksFoucs), len(combinedCircles))
        biconnected_components_dict[tuple(combinedCircles[0])]= [list(it) for it in anyCirclesSum]
        edgeNum= len(biconnectedConp)
        unionLinks= set.union(*anyCirclesSum)
        print("Num of links:", edgeNum)
        print('Num of Detected links:', len(unionLinks))
        print("Circles Detected:", len(anyCirclesSum))
        print("Run loops:", countRun)
        print("m*log(m):", edgeNum*np.log(edgeNum))
        print("Combine biconnected:", len(combinedCircles))# make an algorithm to combine them, minus null space
        print()
    return biconnected_components_dict
    