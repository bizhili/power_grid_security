from skimage.filters import threshold_li
from scipy.stats import chi2
import numpy as np

def li_threshold(residual, groundTruth):
    thresholdLi= threshold_li(residual)
    predict= (residual>thresholdLi)
    cross= groundTruth*predict
    precision= cross.sum()/(predict.sum()+1e-14)
    recall= cross.sum()/(groundTruth.sum()+1e-14)
    cross= (groundTruth)*predict
    falsePositive= cross.sum()/((groundTruth).sum()+1e-14)
    return precision, recall, falsePositive, thresholdLi

def chi_2_threshold(residual, groundTruth, level= 0.95, df= 1, sigma= 1e-3):
    thresholdChi= np.sqrt(chi2.ppf(level, df)*np.square(sigma))
    predict= (residual>thresholdChi)
    cross= groundTruth*predict
    precision= cross.sum()/(predict.sum()+1e-5)
    recall= cross.sum()/(groundTruth.sum()+1e-5)
    cross= (groundTruth)*predict
    falsePositive= cross.sum()/((groundTruth).sum()+1e-5)
    return precision, recall, falsePositive, thresholdChi

def joint_threshold(residual, groundTruth, thresholdLi= None, thresholdChi= None):
    thresholdJoint= max(thresholdLi, thresholdChi)
    predict= (residual>thresholdJoint)
    cross= groundTruth*predict
    precision= cross.sum()/(predict.sum()+1e-5)
    recall= cross.sum()/(groundTruth.sum()+1e-5)
    cross= (groundTruth)*predict
    falsePositive= cross.sum()/((groundTruth).sum()+1e-5)
    return precision, recall, falsePositive, thresholdJoint