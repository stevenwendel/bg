import numpy as np  
import pandas as pd
import psutil
import sys, os
from src.constants import *
from src.neuron import Izhikevich
import pickle   
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def create_alpha_array(length, L=30):
    alphas = [(td / L) * np.exp((L - td) / L) for td in range(1, length+1)]
    rounded_alphas = np.round(alphas, 4)
    return np.array(rounded_alphas)

def alpha_fit(alp_arr, start_time, tmax=5000):
    # Initialize a zero vector of the desired output length
    alpha_padded = np.zeros(tmax)
    
    # Calculate the end time for the alpha array within the padded vector
    end_time = start_time + len(alp_arr)
    
    # Check if the alpha array fits within the output length starting from start_time
    if end_time <= tmax:
        # If it fits, add the entire alpha array to the padded vector
        alpha_padded[start_time:end_time] = alp_arr
    else:
        # If it doesn't fit, truncate the alpha array to fit within the output length
        trunc_size = tmax - start_time
        alpha_padded[start_time:] = alp_arr[:trunc_size]
    
    return alpha_padded


def get_binned_spikes(neuron_spikes: np.ndarray):

    num_neurons = int(len(neuron_spikes))
    binned_spikes = np.array([neu['spike_times'] for neu in neuron_spikes])

    binned_experimental_spikes = None
    binned_control_spikes = None

    binned_experimental_spikes = np.reshape(experimental_spikes, 
                (num_neurons, int(TMAX/BIN_SIZE), BIN_SIZE)
                  ).sum(axis=2)

    binned_control_spikes = np.reshape(control_spikes, 
                (num_neurons, int(TMAX/BIN_SIZE), BIN_SIZE)
                    ).sum(axis=2)

    binned_differences = (binned_experimental_spikes - binned_control_spikes).astype(int)
    return binned_differences



def get_binned_differences(experimental_neurons: dict[str,np.ndarray], #dict of neurons, each w hist_V and spike_times
                           control_neurons: dict[np.ndarray,np.ndarray]):

    assert len(experimental_neurons) == len(control_neurons)


    num_neurons = int(len(experimental_neurons))
    control_spikes = np.array([neu['spike_times'] for neu in control_neurons])
    experimental_spikes = np.array([neu['spike_times'] for neu in experimental_neurons])

    binned_experimental_spikes = None
    binned_control_spikes = None

    binned_experimental_spikes = np.reshape(experimental_spikes, 
                (num_neurons, int(TMAX/BIN_SIZE), BIN_SIZE)
                  ).sum(axis=2)

    binned_control_spikes = np.reshape(control_spikes, 
                (num_neurons, int(TMAX/BIN_SIZE), BIN_SIZE)
                    ).sum(axis=2)

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

# Function to create a DNA string
def create_dna_string(weights, active_synapses):
    # Initialize a DNA list with zeros
    dna = [0] * len(active_synapses)
    
    # Iterate through the weights
    for source, target, weight in weights:
        # Find the index of the connection in ACTIVE_SYNAPSES
        try:
            index = active_synapses.index([source, target])
            # Insert the weight at the found index
            dna[index] = weight
        except ValueError:
            # If the connection is not found, you can choose to ignore or handle it
            print(f"Connection {source} -> {target} not found in ACTIVE_SYNAPSES.")
   
    return dna

def load_ga_run_to_df(file_path: str) -> pd.DataFrame:
    """Load a genetic algorithm run pickle file into a sorted DataFrame.
    
    Args:
        file_path (str): Path to the pickle file containing the GA run data
        
    Returns:
        pd.DataFrame: DataFrame with columns:
            - generation: Generation number
            - dna: DNA sequence as a tuple
            - dna_score: Score for the DNA sequence
        Sorted by dna_score in descending order
    """
    # Load the pickle file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Initialize list to store all rows
    rows = []
    
    # Iterate through each generation
    for key in data.keys():
        if key.startswith('gen_'):
            gen_num = int(key.split('_')[1])
            population = data[key]['population']
            
            # Add each DNA sequence and its score to the rows
            for dna_dict in population:
                rows.append({
                    'generation': gen_num,
                    'dna': tuple(dna_dict['dna']),  # Convert list to tuple for hashability
                    'dna_score': dna_dict['dna_score']
                })
    
    # Create DataFrame and sort
    df = pd.DataFrame(rows)
    df = df.sort_values('dna_score', ascending=False, ignore_index=True)
    
    return df


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB
