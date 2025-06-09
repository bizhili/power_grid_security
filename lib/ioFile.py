import pandas as pd
import numpy as np
from scipy.io import loadmat


def read_excel_power(file_path= "", id1= 0, id2= 1, normalize= False):
    pfData = pd.read_excel(file_path, engine='openpyxl', sheet_name=id1, parse_dates=['Timestamp'])
    phaseData = pd.read_excel(file_path, engine='openpyxl', sheet_name=id2, parse_dates=['Timestamp'])
    try:
        inject = pd.read_excel(file_path, engine='openpyxl', sheet_name=2, parse_dates=['Timestamp'])
        inject.set_index('Timestamp', inplace=True)
        inject= inject.to_numpy(dtype= np.float64)
    except:
        inject= 0
    pfData.set_index('Timestamp', inplace=True)
    phaseData.set_index('Timestamp', inplace=True)
    pfData= pfData.to_numpy(dtype= np.float64)
    phaseData= phaseData.to_numpy(dtype= np.float64)
    if normalize:
        pfData= pfData/np.linalg.norm(pfData, 1, axis=1, keepdims= True)
        pfData= pfData-np.mean(pfData, 0, keepdims= True)
    return pfData, phaseData

def read_mat_power(file_path= "", normalize= True):
    mat_data = loadmat(file_path)
    H = mat_data['Bf']
    pfData = mat_data['measureBranchActivePower']
    qfData = mat_data.get('measureBranchReactivePower', None)
    phaseData = mat_data['stateBusVoltageAngle']
    if normalize:
        pfData= pfData/np.linalg.norm(pfData, 1, axis=1, keepdims= True)
        pfData= pfData-np.mean(pfData, 0, keepdims= True)
    A=np.array(H>0, dtype= np.float32)-np.array(H<0, dtype= np.float32)
    return pfData, phaseData, H, A, qfData

