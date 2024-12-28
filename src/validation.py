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



def score_run(neuron_data: dict[list[Izhikevich],list[Izhikevich]], criteria_dict: dict[dict, dict]) -> int:

    """
    For each CONDITION in neuron_data:
        Use get_neurons to select neurons in CRITERIA_NAMES.
        Compare the spike bins of EACH of the CONDITION neurons to the on/off states in criteria.
        

    """

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



def define_criteria(num_periods):

    tMax=tMax
    num_periods = num_periods

    """Save this criteria as a dictionary a the beginning of the script
    if io = on, then the interval is ON for experimental condition; should be OFF otherwise 
    if io = off, then the interval is OFF for experimental condition; should be ON otherwise 

    """

    CRITERIA = {
        # These are all intervals which should be ON for experimental condition; should be OFF otherwise 
        "experimental_criterion" : {
            "Somat": {
                "interval":[EPOCHS['sample'][0], EPOCHS['sample'][1]],
                "io": "on"
            },
            "ALMprep": {
                "interval":[EPOCHS['sample'][0], EPOCHS['delay'][1]],
                "io": "on"
            },
            "ALMinter": {
                "interval":[EPOCHS['response'][0], EPOCHS['response'][0] + 300],
                "io": "on"
            },
            "ALMresp": {
                "interval":[EPOCHS['response'][0], EPOCHS['response'][1]], #tMax -250?
                "io": "on"
            },
            "SNR1": {
                "interval":[EPOCHS['sample'][0], EPOCHS['delay'][1]],
                "io": "off"
            },
            "SNR2": {   
                "interval":[EPOCHS['response'][0], TMAX-250],
                "io": "off"
            },
            "VMprep": {
                "interval":[EPOCHS['sample'][0], EPOCHS['delay'][1]],
                "io": "on"
            },
            "VMresp": {
                "interval":[EPOCHS['response'][0], TMAX-250],
                "io": "on"
            },
            "PPN": {
                "interval":[EPOCHS['response'][0], EPOCHS['response'][0]+250],
                "io": "on"
            } 
        },
        "control_criterion" : {
            "Somat": {
                "interval":[EPOCHS['sample'][0], EPOCHS['sample'][1]],
                "io": "off"
            },
            "ALMinter": {
                "interval":[EPOCHS['response'][0], EPOCHS['response'][0] + 300],
                "io": "off"
            },
            "ALMresp": {
                "interval":[EPOCHS['response'][0], EPOCHS['response'][1]], #tMax -250?
                "io": "off"
            },
            "SNR1": {
                "interval":[0,TMAX],
                "io": "on"
            },
            "SNR2": {   
                "interval":[0,TMAX],
                "io": "on"
            },
            "VMprep": {
                "interval":[EPOCHS['sample'][0], EPOCHS['delay'][1]],
                "io": "off"
            },
            "VMresp": {
                "interval":[EPOCHS['response'][0], TMAX-250],
                "io": "off"
            },
            "PPN": {
                "interval":[EPOCHS['response'][0], EPOCHS['response'][0]+250],
                "io": "on"
            }
        }
    }

    crit_experimental_df = pd.DataFrame(CRITERIA['experimental_criterion'])
    crit_control_df = pd.DataFrame(CRITERIA['control_criterion'])

    # Experiment criteria by epoch
    experiment_criteria_by_epoch = np.array([
        [0, 1, 0, 0, 0],  # Somat
        [0, 1, 1, 0, 0],  # ALMprep
        [0, 0, 0, 0, 0],  # ALMinter
        [0, 0, 1, 0, 0],  # ALMresp
        [0, 1, 1, 0, 0],  # SNR1
        [0, 0, 0, 1, 1],  # SNR2
        [1, 1, 0, 0, 0],  # VMprep
        [0, 0, 0, 1, 1],  # VMresp
        [0, 0, 0, 0, 0]   # PPN
    ])

    # Control criteria by epoch
    control_criteria_by_epoch = np.array([
        [0, 0, 0, 0, 0],  # Somat
        [0, 0, 0, 0, 0],  # ALMprep
        [0, 0, 0, 1, 0],  # ALMinter
        [0, 0, 0, 0, 0],  # ALMresp
        [1, 1, 1, 1, 1],  # SNR1
        [1, 1, 1, 1, 1],  # SNR2
        [0, 0, 0, 0, 0],  # VMprep
        [0, 0, 0, 0, 0],  # VMresp
        [0, 0, 0, 0, 0]   # PPN
    ])
    num_epochs = experiment_criteria_by_epoch.shape[1]

    broadcasted_difference = np.repeat(
        experiment_criteria_by_epoch - control_criteria_by_epoch, 
        repeats=num_periods/num_epochs, 
        axis=1
    )
    return broadcasted_difference



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