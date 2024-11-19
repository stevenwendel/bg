import numpy as np
import pandas as pd
from src.constants import *

# def is_firing(neu, window, threshold, validation_state):
#     """
#     Validates the firing rate of a neuron within a specified window.
    
#     Parameters:
#         neu (Izhikevich): The neuron instance to validate.
#         window (list of int): The time window [start, end] for validation.
#         threshold (float): The firing rate threshold.
#         validation_state (bool): The expected state (True for active, False for inactive).
    
#     Returns:
#         bool: True if validation passes, False otherwise.
#     """
#     firing_rate = sum(neu.spike_times[window[0]:window[1]]) / (window[1] - window[0]) * 1000
#     return (validation_state is None) or ((firing_rate > threshold) == validation_state)

# def score_neuron(neu, criteria, validation_thresholds, validation_period_times):
#     """
#     Scores a single neuron based on the provided criteria.
    
#     Parameters:
#         neu (Izhikevich): The neuron instance to score.
#         criteria (pd.DataFrame): The validation criteria DataFrame.
#         validation_thresholds (list of float): Thresholds for each neuron.
#         validation_period_times (list of list of int): Time windows for each validation period.
    
#     Returns:
#         tuple: (total_score, neuron_period_scores)
#     """
#     validation_matrix = criteria.to_numpy()
#     neu_index = criteria.index.tolist().index(neu.name)
#     neuron_period_scores = []
#     for i, period in enumerate(validation_period_times):
#         result = is_firing(
#             neu,
#             period,
#             validation_thresholds[neu_index],
#             validation_matrix[neu_index][i]
#         )
#         neuron_period_scores.append(result)
#     total_score = sum(neuron_period_scores)
#     return total_score, neuron_period_scores

def score_run(binned_differences_df: pd.DataFrame, diff_criteria_df: pd.DataFrame):
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
            expected = True if abs(criteria_ts[period]) > 0 else False
            active = neuron_ts[period] > 0
            if active and expected:
                score += 1
            elif not active and not expected:
                score += 1
            else:
                score -= 1
    return score

def define_criteria(num_periods):
    """
    Defines the experiment and control criteria based on epochs and subdivisions.
    
    Parameters:
        epochs (dict): Dictionary defining epoch names and their time ranges.
        num_periods (int): Number of subdivisions per epoch.
        validation_neurons (list of Izhikevich): List of neurons to validate.
        my_free_weights_names (list of str): Names of the free weights.
    
    Returns:
        tuple: (experiment_criteria, control_criteria, validation_period_times, validation_period_names)
    """
    
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
    # Define validation period times and names
    validation_period_times = []
    validation_period_names = []
    for epoch_name, (start, end) in epochs.items():
        epoch_duration = end - start
        period_duration = epoch_duration / number_of_subdivisions
        for subdivision in range(number_of_subdivisions):
            period_start = int(start + subdivision * period_duration)
            period_end = int(period_start + period_duration)
            validation_period_times.append([period_start, period_end])
            validation_period_names.append(f'{epoch_name[0]}{subdivision + 1}')  # e.g., 's1', 's2', etc.
    
    # Create criteria DataFrames
    def create_criteria_df(criteria_by_epoch):
        master_list = []
        for neuron_criteria in criteria_by_epoch:
            neuron_divided_criteria = []
            for epoch in neuron_criteria:
                for _ in range(number_of_subdivisions):
                    neuron_divided_criteria.append(epoch)
            master_list.append(neuron_divided_criteria)
        return pd.DataFrame(master_list, index=[neu.name for neu in validation_neurons], columns=validation_period_names)
    
    experiment_criteria = create_criteria_df(experiment_criteria_by_epoch)
    control_criteria = create_criteria_df(control_criteria_by_epoch)
    
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