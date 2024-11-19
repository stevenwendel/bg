import numpy as np  
import pandas as pd
from src.constants import *
from src.neuron import Izhikevich
import pickle   
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def create_alpha_array(length, L=30):
    alphas = [(td / L) * np.exp((L - td) / L) for td in range(1, length + 1)]
    return np.array(alphas)

def alpha_fit(alp_arr, time, time_max):
    alpha_padded = np.zeros(time_max)
    if time_max >= len(alp_arr) + time:
        alpha_padded[time:time + len(alp_arr)] += alp_arr
    else:
        trunc_size = time_max - time
        alpha_padded[time:] += alp_arr[:trunc_size]
    return alpha_padded


def get_binned_differences(experimental_neurons, control_neurons):
    assert len(experimental_neurons) == len(control_neurons)

    num_neurons = int(len(experimental_neurons))
    control_spikes = np.array([neu.spike_times for neu in control_neurons])
    experimental_spikes = np.array([neu.spike_times for neu in experimental_neurons])

    binned_experimental_spikes = None
    binned_control_spikes = None

    for i, set in enumerate([experimental_spikes, control_spikes]):
        binned_spikes = np.reshape(set, 
                (num_neurons, int(TMAX/BIN_SIZE), BIN_SIZE)
                    ).sum(axis=2)
        # print(f'{binned_spikes=}')
        
        if i == 0:
            binned_experimental_spikes = binned_spikes
        else:
            binned_control_spikes = binned_spikes

    binned_differences = (binned_experimental_spikes - binned_control_spikes).astype(int)
    return binned_differences


def get_neurons(neuron_data, target_neurons):
    df=pd.DataFrame(neuron_data, index=NEURON_NAMES)

    # Find the indices in NEURON_NAMES that match criteria_names
    matching_indices = [i for i, name in enumerate(NEURON_NAMES) if name in target_neurons]
    
    # Use these indices to filter the DataFrame
    filtered_df = df.iloc[matching_indices]
    
    # Convert the filtered DataFrame back to a NumPy array
    return filtered_df.to_numpy()


def save_neurons(neurons: list[Izhikevich], condition):
    with open(f"./data/{condition}_neurons.pkl", "wb") as f:
        pickle.dump(neurons, f)

# Probably broken and need to fix, but later.
def load_neurons(file_path: str):
    with open(file_path,'rb') as f:
        try:
            while True:
                # Load each (dna_score, dna_0) pair
                dna_score, test_dna = pickle.load(f)
                print(f'Loaded dna_score: {dna_score}, dna_0: {test_dna}')
        except EOFError:
            # End of file reached
            pass