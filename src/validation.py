import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.neuron import Izhikevich
from src.constants import *



# def score_run(binned_differences_df: pd.DataFrame, diff_criteria_df: pd.DataFrame):
#     """Scores how well neural activity matches expected criteria.

#     Compares binned firing rate differences between experimental and control conditions
#     against expected differences defined in criteria. Awards points when activity matches
#     expectations and penalizes mismatches.

#     Args:
#         binned_differences_df (pd.DataFrame): Observed firing rate differences between 
#             experimental and control conditions. Each row is a neuron, columns are time periods.
#         diff_criteria_df (pd.DataFrame): Expected differences between conditions.
#             Must have same shape as binned_differences_df. Positive values indicate expected
#             higher firing in experimental condition.

#     Returns:
#         int: Score indicating how well activity matched criteria. Higher is better.
#             +1 for each match (active when expected or inactive when not expected)
#             -1 for each mismatch (active when not expected or inactive when expected)

#     Example:
#         >>> differences = pd.DataFrame([[1,0], [0,-1]])  # 2 neurons, 2 time periods
#         >>> criteria = pd.DataFrame([[1,0], [0,0]])      # Expect activity only in [0,0]
#         >>> score = score_run(differences, criteria)
#         >>> assert score == 3  # 3 matches, 1 mismatch
#     """
#     if binned_differences_df is not pd.DataFrame:
#         binned_differences_df = pd.DataFrame(binned_differences_df)
#     if diff_criteria_df is not pd.DataFrame:
#         diff_criteria_df = pd.DataFrame(diff_criteria_df)
     
#     assert (binned_differences_df.shape == diff_criteria_df.shape), "Shapes are incongruent"

#     score = 0
#     for neu in diff_criteria_df.index:
#         neuron_ts = binned_differences_df.loc[neu]
#         criteria_ts = diff_criteria_df.loc[neu]
#         # print(f'{neuron_ts=}')
#         # print(f'{criteria_ts=}')    
#         for period in criteria_ts.index:
#             expected = True if criteria_ts[period] > 0 else False
#             active = abs(neuron_ts[period]) > 0
#             if active and expected:
#                 score += 1
#             elif not active and not expected:
#                 score += 1
#             else:
#                 score -= 1
#     return score


def define_criteria(num_periods: int) -> dict[str, np.ndarray]:

    assert len(CRITERIA['experimental']) == len(CRITERIA['control']) == len(CRITERIA_NAMES), "Criteria for experimental and control conditions must be the same length"

    tMax = TMAX
    num_criteria = len(CRITERIA_NAMES)
    period_size = TMAX // num_periods

    criteria = {}

    for condition in ['experimental', 'control']:
    
        io_matrix = np.zeros((num_criteria, num_periods))
        io_dict = CRITERIA[condition]

        for i, (key, value) in enumerate(io_dict.items()):
            interval = value['interval']
            io = value['io']
            start_period = interval[0] // period_size
            end_period = min(interval[1] // period_size, num_periods)
            io_matrix[i, start_period:end_period] = 1 if io == 'on' else 0
        criteria[condition] = io_matrix

    """Save this criteria as a dictionary a the beginning of the script
    if io = on, then the interval is ON for experimental condition; should be OFF otherwise 
    if io = off, then the interval is OFF for experimental condition; should be ON otherwise 

    """
    return criteria


def calculate_score(matrix1, matrix2):
    # Ensure matrices have the same dimensions
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        raise ValueError("Matrices must have the same dimensions.")

    score = 0

    # Loop through both matrices
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            # Award a point if both entries are 0 or both are positive
            if (matrix1[i][j] == 0 and matrix2[i][j] == 0) or (matrix1[i][j] > 0 and matrix2[i][j] > 0):
                score += 1
                # Special case scoring
                if i==7 and 30<j<40:
                    score+=3
    
    # First attempt at applying L1 norm... 
    # Could use this to calcualte score for matrices... 
    # l1_norm = norm(matrix1 - matrix2, ord=1)

    # Gonna put this in the GA.py script... 
    # score -= l1_norm

    return score

""" ====deprecated====

def score_run(
        target_binned_differences: dict[str,np.ndarray], #Receiving target_binned_differences
        io_criteria: np.ndarray) -> int: # 9x20 matrix (i.e. array of arrays) of 0s and 1s
    
    #check to see what i'm looking at
    print(target_binned_differences)
    print(io_criteria)

    raise Exception("Not implemented")
    
    # For each CONDITION in neuron_data:
    #     Use get_neurons to select neurons in CRITERIA_NAMES.
    #     Compare the spike bins of EACH of the CONDITION neurons to the on/off states in criteria.
    

    if binned_differences_df is not pd.DataFrame:
        binned_differences_df = pd.DataFrame(binned_differences_df)
    if diff_criteria_df is not pd.DataFrame:
        diff_criteria_df = pd.DataFrame(diff_criteria_df)
     
    assert (binned_differences_df.shape == diff_criteria_df.shape), "Shapes are incongruent"

    score = 0
    for neu in diff_criteria_df.index:
        neuron_ts = binned_differences_df.loc[neu]
        criteria_ts = diff_criteria_df.loc[neu]
        # print(f'{neuron_ts=}')
        # print(f'{criteria_ts=}')    
        for period in criteria_ts.index:
            expected = True if criteria_ts[period] > 0 else False
            active = abs(neuron_ts[period]) > 0
            if active and expected:
                score += 1
            elif not active and not expected:
                score += 1
            else:
                score -= 1
    return score



"""
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