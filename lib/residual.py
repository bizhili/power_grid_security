import numpy as np
import lib.cycle_space as csp

def cycle_residual(trainDataNp, rankH, m, testData, cycleList= None, sigma= None):
    predParas, cSpace, wholeRight= csp.get_cycle_space(trainDataNp, rankH, m, cycleList= cycleList)
    # cProjection= cSpace@np.linalg.inv(cSpace.T@cSpace)@cSpace.T #cSpace@np.linalg.inv(cSpace.T@cSpace)@cSpace.T; cSpace.T
    cProjection= cSpace.T
    ErrorC= cProjection@(testData.T)
    if sigma is not None:
        ErrorC= np.square(ErrorC)/sigma
        pass
    residualC= np.abs(ErrorC)
    return residualC, cSpace, wholeRight

def svd_residual(trainDataNp, rankH, testData, wholeRight= None):
    pfDatanor2= testData#/np.linalg.norm(testData, 2, axis=1, keepdims= True)
    pfDataSdr2= pfDatanor2 #- np.mean(pfDatanor2, 0, keepdims= True)
    if wholeRight is None:
        wholeRight= trainDataNp.T.dot(trainDataNp)
    m= wholeRight.shape[0]
    eigvals, eigvecs = np.linalg.eigh(wholeRight)
    nullH = eigvecs[:, :m-rankH]
    # hProjection= nullH@np.linalg.inv(nullH.T@nullH)@nullH.T
    ErrorH= nullH.T@(pfDataSdr2.T)
    residualH= np.linalg.norm(ErrorH, axis= 0)
    return residualH, nullH

def true_residual(H, m, testData):
    TPinv= np.linalg.pinv(H)
    testT= np.eye(m)-H@TPinv
    ErrorT= testT@(testData.T)
    residualT= np.linalg.norm(ErrorT, axis= 0)
    return residualT