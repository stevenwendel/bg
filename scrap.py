import pickle
from src import *
from src.neuron import Izhikevich
import numpy as np
import pandas as pd



"""
# with open('./data/experimental_neurons.pkl', 'rb') as f:
#     experimental_neurons = pickle.load(f)

# with open('./data/control_neurons.pkl', 'rb') as f:
#     control_neurons = pickle.load(f)

# assert len(experimental_neurons) == len(control_neurons)

# binned_differences = get_binned_differences(experimental_neurons, control_neurons, 100)
# plot_binned_differences(binned_differences, 100)

# Convert lists to NumPy arrays
experiment_criteria_by_epoch = pd.Series({
    "Somat": [0, 1, 0, 0, 0],
    "ALMprep": [0, 1, 1, 0, 0],
    "ALMinter": [0, 0, 0, 0, 0],
    "ALMresp": [0, 0, 1, 0, 0],
    "SNR1": [0, 1, 1, 0, 0],
    "SNR2": [0, 0, 0, 1, 1],
    "VMprep": [1, 1, 0, 0, 0],
    "VMresp": [0, 0, 0, 1, 1],
    "PPN": [0, 0, 0, 0, 0]
})

# Define criteria by epoch for control
control_criteria_by_epoch = pd.Series({
    "Somat": [0, 0, 0, 0, 0],
    "ALMprep": [0, 0, 0, 0, 0],
    "ALMinter": [0, 0, 0, 1, 0],
    "ALMresp": [0, 0, 0, 0, 0],
    "SNR1": [1, 1, 1, 1, 1],
    "SNR2": [1, 1, 1, 1, 1],
    "VMprep": [0, 0, 0, 0, 0],
    "VMresp": [0, 0, 0, 0, 0],
    "PPN": [0, 0, 0, 0, 0]
})

# Element-wise subtraction
difference_criteria_by_epoch = experiment_criteria_by_epoch.combine(
    control_criteria_by_epoch, 
    lambda x, y: np.subtract(x, y)
)

print(difference_criteria_by_epoch)
"""