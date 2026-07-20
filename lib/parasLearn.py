import copy
import numpy as np

def paras_learning(biconnected_components, wholeCov, originalData):
    m= wholeCov.shape[1]
    predParas= np.ones((m, 1))
    biconnected_components_dictCp= copy.deepcopy(biconnected_components) 
    cSpace= []
    for biconnected in biconnected_components_dictCp:
        sub_cycles_with_edges= biconnected.copy()
        # print(len(sub_cycles_with_edges))
        if len(sub_cycles_with_edges)==0:
            continue
        line_parameters= []
        for item in sub_cycles_with_edges:
            # submeasurement= originalData[:, item]
            covM= wholeCov[:, list(item)][list(item), :]
            U, S, V= np.linalg.svd(covM)
            # print(S[-1])
            UT= V[-1, :][:, None]

            # Val, Vt = np.linalg.eig(covM)
            # Val= np.abs(np.real(Val))
            # minIdx= np.argmin(Val)
            # UT= Vt[:, minIdx:minIdx+1]

            line_parameter= UT[:, -1]
            cVector= np.zeros(m)
            cVector[list(item)]= UT[:, -1]
            cSpace.append(cVector)
            line_parameters.append(np.abs(line_parameter))
        while(len(sub_cycles_with_edges)>1):
            mainTree= sub_cycles_with_edges[0].copy()
            for thisTreeI, item in enumerate(sub_cycles_with_edges):
                if 1-set(mainTree).isdisjoint(set(item)) and thisTreeI!=0:
                    break
            thisTree= sub_cycles_with_edges[thisTreeI].copy()
            common_indices = [(i, thisTree.index(val)) for i, val in enumerate(mainTree) if val in thisTree]
            # common_index= common_indices[0]
            ratio= 0
            for common_index in common_indices:
                ratio+= line_parameters[0][common_index[0]]/line_parameters[thisTreeI][common_index[1]]
            ratio= ratio/len(common_indices)
            line_parameters[thisTreeI]= line_parameters[thisTreeI]*ratio
            delIndex= {delIndex[1] for delIndex in common_indices}
            thisTreeWeight= np.delete(line_parameters[thisTreeI], list(delIndex))
            for j in range(len(thisTree)):
                if j not in delIndex:
                    sub_cycles_with_edges[0].append(thisTree[j])
            line_parameters[0]= np.concatenate([line_parameters[0][:, None], thisTreeWeight[:, None]]).squeeze()
            sub_cycles_with_edges.pop(thisTreeI)
            line_parameters.pop(thisTreeI)
        for i, item in enumerate(sub_cycles_with_edges[0]):
            predParas[item]= 1/line_parameters[0][i]
    return predParas, np.stack(cSpace).T

def para_precision(H, predParas):
    P= np.sum(np.abs(H), axis= 1, keepdims= True)/2
    print(np.std((np.abs(predParas)/P)[0:10]))