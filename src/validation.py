import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from src.neuron import Izhikevich
from src.constants import *


def define_criteria() -> dict[str, np.ndarray]:

    assert len(CRITERIA['experimental']) == len(CRITERIA['control']) == len(CRITERIA_NAMES), "Criteria for experimental and control conditions must be the same length"

    tMax = TMAX
    num_periods = TMAX // BIN_SIZE
    criteria = {}

    for condition in ['experimental', 'control']:
    
        io_matrix = np.zeros((len(CRITERIA_NAMES), num_periods))
        
        # Set rows to 1 for names in the TONICALLY_ACTIVE list
        for i, name in enumerate(CRITERIA_NAMES):
            if name in TONICALLY_ACTIVE_NEURONS:
                io_matrix[i, :] = 1  # Set the entire row to 1

        io_dict = CRITERIA[condition]

        for i, (key, value) in enumerate(io_dict.items()):
            interval = value['interval']
            io = value['io']
            start_period = interval[0] // BIN_SIZE
            end_period = min(interval[1] // BIN_SIZE, num_periods)
            io_matrix[i, start_period:end_period] = 1 if io == 'on' else 0
        criteria[condition] = io_matrix

    """Save this criteria as a dictionary a the beginning of the script
    if io = on, then the interval is ON for experimental condition; should be OFF otherwise 
    if io = off, then the interval is OFF for experimental condition; should be ON otherwise 

    """
    return criteria


def calculate_score(critSpikeMatrix, critCriteriaMatrix, condition):
    # Ensure matrices have the same dimensions
    if len(critSpikeMatrix) != len(critCriteriaMatrix) or len(critSpikeMatrix[0]) != len(critCriteriaMatrix[0]):
        raise ValueError("Matrices must have the same dimensions.")

    score = 0

    # Loop through both matrices
    for i in range(len(critSpikeMatrix)):
        for j in range(len(critSpikeMatrix[0])):
            # Award a point if both entries are 0 or both are positive
            if (critSpikeMatrix[i][j] == 0 and critCriteriaMatrix[i][j] == 0) or (critSpikeMatrix[i][j] > 0 and critCriteriaMatrix[i][j] > 0):
                score += 1
            # else:
            #     print(f'Neuron#: {CRITERIA_NAMES[i]} ==== Period#: {j} ==== Time: {j*BIN_SIZE}-{(j+1)*BIN_SIZE} ==== Condition: {condition}')
                # print(f'Spikes: {int(critSpikeMatrix[i][j])} ==== Criteria: {int(critCriteriaMatrix[i][j])}')
            #     if (CRITERIA_NAMES[i] == 'ALMresp') and j==1:
            #         pass
    
    # First attempt at applying L1 norm... 
    # Could use this to calcualte score for matrices... 
    # l1_norm = norm(matrix1 - matrix2, ord=1)

    # Gonna put this in the GA.py script... 
    # score -= l1_norm

    if condition == 'control':
        score *= 0.5

    return int(score)

"""
    # NEED TO REIMPLEMENT THESE LITTLE CASES.
    # Hard to do because the number of periods is not pre-defined, so I don't know what part to look at.
    # Might be easiest to just SET the number of subdivisions, and 
    # then update diff-matrix to manually program which period should be 1/0.
    # This will break (quietly) as soon as NUM_SUBDIVISIONS is changed....
    # Could add an assert function to protect.
    # Could use a range-finding function to figure out which interval should be updated. 
    # Simple math + floor function +ceiling + lookup should do.
    # i.e. for t=3000 to t=3400, divide each by period_duration, then cieling (20)/floor(22), then access in periods[:,20:22] = 1 
    
    experiment_criteria.loc["PPN", "s1"] = 1
    experiment_criteria.loc["ALMinter", "s1"] = 1
    experiment_criteria.loc["ALMresp", "s3"] = 0
    
    control_criteria.loc["PPN", "s1"] = 1
    control_criteria.loc["ALMinter", "s1"] = 1
    control_criteria.loc["ALMresp", "s3"] = 0
"""