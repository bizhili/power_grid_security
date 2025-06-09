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