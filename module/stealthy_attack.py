
import numpy as np

def extract_null_state(HMat):
    _, S, V= np.linalg.svd(HMat)
    stateNullSpace= V[-1, ]
    return stateNullSpace


def create_attack(HMat, stateNullSpace, Scale):
    return stateNullSpace*Scale, HMat.dot(stateNullSpace*Scale)


def inject_attack_to_state(normalState, stateAttack, startIdx, endIdx):
    attackState= normalState.copy()
    attackState[startIdx: endIdx, :]+= stateAttack
    return attackState

def inject_attack_to_measure(normalMeasure, MeasureAttack, startIdx, endIdx):
    attackMeasure= normalMeasure.copy()
    attackMeasure[startIdx: endIdx, :]+= MeasureAttack
    return attackMeasure
