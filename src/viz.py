import plotly.graph_objs as go
from plotly.subplots import make_subplots
from IPython.display import display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.constants import *

def plot_neurons(neurons, sq_wave, go_wave):
    fig, axs = plt.subplots(len(neurons), 1, figsize=(6, 3 * len(neurons)))
    for i, neu in enumerate(neurons):
        axs[i].plot(range(TMAX), neu.hist_V, label="V")
        axs[i].plot(range(TMAX), sq_wave, label="SqWave", alpha=0.8, color="red", linestyle="dotted")
        axs[i].plot(range(TMAX), go_wave / 5, label="GoWave", alpha=0.8, color="red", linestyle="dotted")
        axs[i].set_title(f"{neu.name} dynamics")
        axs[i].set_xlabel("ms")
        axs[i].set_ylabel("mV")
        axs[i].grid(True)
        axs[i].legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def display_matrix(matrix, nodes):
    assert matrix.shape == (len(nodes), len(nodes)), "Weight Matrix must be the same rank as the neuron name vector"
    df = pd.DataFrame(matrix, columns=nodes, index=nodes)
    display(df) 
    
def plot_neurons_interactive(hist_Vs, neuron_names, sq_wave, go_wave, show_u=False):
    assert len(hist_Vs) == len(neuron_names), "Must have the same number of neurons as the number of hist_Vs"
    n_neurons = len(neuron_names)
    n_cols = 1  # Set to 1 for a single column layout
    n_rows = n_neurons  # Each neuron gets its own row

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=neuron_names)

    hover_template = 'Time: %{x} ms<br>Value: %{y} mV'
    v_color = 'blue'  # Define a consistent color for hist_V

    for i, hist_V in enumerate(hist_Vs):
        row = i + 1  # Adjust row index for single column
        col = 1
        fig.add_trace(go.Scatter(x=list(range(TMAX)), y=hist_V, mode='lines', name='V',
                                 line=dict(color=v_color),  # Use the consistent color
                                 hovertemplate=hover_template), row=row, col=col)
        # if show_u:
        #     fig.add_trace(go.Scatter(x=list(range(TMAX)), y=neu.hist_u, mode='lines', name='u',
        #                              line=dict(dash='dot', color='red'), hovertemplate=hover_template), row=row, col=col)
        fig.add_trace(go.Scatter(x=list(range(TMAX)), y=sq_wave, mode='lines', name='SqWave',
                                 line=dict(dash='dot', color='red'), hovertemplate=hover_template), row=row, col=col)
        fig.add_trace(go.Scatter(x=list(range(TMAX)), y=go_wave / 5, mode='lines', name='GoWave',
                                 line=dict(dash='dot', color='red'), hovertemplate=hover_template), row=row, col=col)

    fig.update_layout(height=300 * n_rows, width=900, title_text="Neuron Dynamics", showlegend=False)
    fig.show()


def plot_binned_differences(binned_differences):
    tMax = int(len(binned_differences[0]) * BIN_SIZE)
    time_intervals = np.arange(0, TMAX, BIN_SIZE)  # Create time intervals for the x-axis

    n_neurons = int(len(binned_differences))
    n_cols = 1  # Set to 1 for a single column layout
    n_rows = n_neurons  # Each neuron gets its own row

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[name for name in NEURON_NAMES])

    hover_template = 'Time: %{x} ms<br>Value: %{y} spikes'
    v_color = 'orange'  # Define a consistent color for spike times

    # it would be nice to have 3 columns to show the experimental, control, and difference

    for i, neu in enumerate(binned_differences):
        row = i + 1  
        col = 1
        fig.add_trace(go.Bar(
            x=time_intervals, 
            y=neu, 
            name=f'Neuron {i+1}',
            marker=dict(color=v_color),  
            hovertemplate=hover_template), 
            row=row, 
            col=col
            )


        # Calculate the maximum absolute value for symmetric y-axis for each neuron
        max_abs_value = max(abs(neu.min()), abs(neu.max()))

        # Update y-axis range for each subplot
        fig.update_yaxes(range=[-max_abs_value, max_abs_value], row=row, col=col)

    fig.update_layout(
        height=300 * n_rows, 
        width=900, 
        title_text="Binned spikes: Experimental - Control", 
        showlegend=False, 
        bargap=0
    )
    fig.show()