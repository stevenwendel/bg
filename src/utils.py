import numpy as np  
import pandas as pd
from src.constants import *
import pickle   
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def load_dna(all_neurons, free_weights_list, dna):
    assert len(free_weights_list)==len(dna), "Number of available synapses does not match length of DNA"

    w = np.zeros((len(all_nodes), len(all_nodes)))
    for i, synapse in enumerate(free_weights_list):
        origin, termina = synapse
        
        origin_index = all_nodes.index(origin)
        termina_index = all_nodes.index(termina)
        w[origin_index, termina_index] = dna[i]
    return w

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


def get_binned_differences(experimental_neurons, control_neurons, bin_size=100):
    assert len(experimental_neurons) == len(control_neurons)

    num_neurons = int(len(experimental_neurons))
    bin_size = bin_size
    tMax = int(len(control_neurons[0].spike_times))

    control_spikes = np.array([neu.spike_times for neu in control_neurons])
    experimental_spikes = np.array([neu.spike_times for neu in experimental_neurons])

    # print(f'{experimental_spikes.shape=}')
    # print(f'{control_spikes.shape=}')

    binned_experimental_spikes = None
    binned_control_spikes = None

    for i, set in enumerate([experimental_spikes, control_spikes]):
        binned_spikes = np.reshape(set, 
                (num_neurons, int(tMax/bin_size), bin_size)
                    ).sum(axis=2)
        # print(f'{binned_spikes=}')
        
        if i == 0:
            binned_experimental_spikes = binned_spikes
        else:
            binned_control_spikes = binned_spikes

    binned_differences = (binned_experimental_spikes - binned_control_spikes).astype(int)
    return binned_differences

def plot_binned_differences(binned_differences, bin_size):
    tMax = int(len(binned_differences[0]) * bin_size)
    time_intervals = np.arange(0, tMax, bin_size)  # Create time intervals for the x-axis

    n_neurons = int(len(binned_differences))
    n_cols = 1  # Set to 1 for a single column layout
    n_rows = n_neurons  # Each neuron gets its own row

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[name for name in all_nodes])

    hover_template = 'Time: %{x} ms<br>Value: %{y} spikes'
    v_color = 'orange'  # Define a consistent color for spike times

    # it would be nice to have 3 columns to show the experimental, control, and difference

    for i, neu in enumerate(binned_differences):
        row = i + 1  
        col = 1
        fig.add_trace(go.Bar(x=time_intervals, y=neu, name=f'Neuron {i+1}',
                            marker=dict(color=v_color),  
                            hovertemplate=hover_template), row=row, col=col)

        # Calculate the maximum absolute value for symmetric y-axis for each neuron
        max_abs_value = max(abs(neu.min()), abs(neu.max()))

        # Update y-axis range for each subplot
        fig.update_yaxes(range=[-max_abs_value, max_abs_value], row=row, col=col)

    fig.update_layout(
        height=300 * n_rows, 
        width=900, 
        title_text="Exp-Control Spikes", 
        showlegend=False, 
        bargap=0
    )
    fig.show()