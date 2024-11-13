import numpy as np
import matplotlib.pyplot as plt
from src.constants import *
from src.neuron import *
from src.utils import *
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from IPython.display import display

""" Where is my weight matrix and default neurons? I should create a scratch and use that."""
def create_experiment(tMax, bin_size=100):

    n_bins = tMax / bin_size + 1
    periods = np.linspace(0, tMax, int(n_bins))

    # Creating wave inputs
    sqWave = np.zeros(tMax)
    goWave = np.zeros(tMax)
    sqWave[epochs['sample'][0]:epochs['sample'][1]] = 145.
    goWave[epochs['response'][0]:epochs['response'][0] + go_signal_duration] = 850.
    input_waves = [sqWave,goWave]

    # Create fixed alpha array of length 250
    alphaArray = create_alpha_array(250, L=30)

    return periods, input_waves, alphaArray

def create_neurons(neuron_names: list[str]) ->list[Izhikevich]:

    # Instantiating neurons
    neurons = []
    for neu in neuron_names:
        if neu in ["MSN1", "MSN2", "MSN3"]:
            neuron_instance = Izhikevich(name = neu, neuron_type="msn")
        else:
            neuron_instance = Izhikevich(name = neu, neuron_type="rs")
        globals()[neu] = neuron_instance # Makes instances available globally
        neurons.append(neuron_instance) # Creates a list of all Iz neurons; note, these are the actual objects, not a list of names!

    SNR1.E = SNR2.E = SNR3.E = 112.0 
    PPN.E = 100.0 
    return neurons

def run_network(weight_matrix, neurons, sq_wave, go_wave, t_max, dt, alpha_array, control=False):
    # Initializing neuron records
    for neu in neurons:
        neu.restart()
        neu.input = np.zeros(t_max)
        neu.spike_times = np.zeros(t_max)
        neu.hist_V = np.zeros(t_max)
        neu.hist_u = np.zeros(t_max)
        neu.hist_V[0] = neu.V
        neu.hist_u[0] = neu.u

        if neu.name == "Somat" and not control:
            neu.input += sq_wave
        elif neu.name == "PPN":
            neu.input += go_wave

    """
    # Setting weights
    w = weight_matrix.copy()
    if control:
        w[0, 4] = 0  # Somat_ALMprep=0
        w[0, 1] = 0  # Somat_MSN1=0
    """
    w=weight_matrix

    spikers = []

    # Running the network
    for t in range(1, t_max, dt):
        # Distributing alphas
        if spikers:
            alpha = alpha_fit(alpha_array, t - 1, t_max)
            for i, post in enumerate(neurons):
                post.input += alpha * np.dot(spikers, w)[i]

        # Updating V and u, and collecting spikes
        spikers = []
        for neu in neurons:
            neu.hist_V[t], neu.hist_u[t], neu.spike_times[t - 1] = neu.update(dt=dt, I_ext=neu.input[t], sigma=0)
            spikers.append(neu.spiked)
            if neu.spiked:
                neu.hist_V[t - 1] = neu.vpeak

def plot_neurons(neurons, sq_wave, go_wave, t_max):
    fig, axs = plt.subplots(len(neurons), 1, figsize=(6, 3 * len(neurons)))
    for i, neu in enumerate(neurons):
        axs[i].plot(range(t_max), neu.hist_V, label="V")
        axs[i].plot(range(t_max), sq_wave, label="SqWave", alpha=0.8, color="red", linestyle="dotted")
        axs[i].plot(range(t_max), go_wave / 5, label="GoWave", alpha=0.8, color="red", linestyle="dotted")
        axs[i].set_title(f"{neu.name} dynamics")
        axs[i].set_xlabel("ms")
        axs[i].set_ylabel("mV")
        axs[i].grid(True)
        axs[i].legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def display_matrix(matrix, nodes):
    df = pd.DataFrame(matrix, columns=nodes, index=nodes)
    display(df) #Does this work?

def plot_neurons_interactive(neurons, sq_wave, go_wave, t_max, show_u=False):
    n_neurons = len(neurons)
    n_cols = 1  # Set to 1 for a single column layout
    n_rows = n_neurons  # Each neuron gets its own row

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[neu.name for neu in neurons])

    hover_template = 'Time: %{x} ms<br>Value: %{y} mV'
    v_color = 'blue'  # Define a consistent color for hist_V

    for i, neu in enumerate(neurons):
        row = i + 1  # Adjust row index for single column
        col = 1
        fig.add_trace(go.Scatter(x=list(range(t_max)), y=neu.hist_V, mode='lines', name='V',
                                 line=dict(color=v_color),  # Use the consistent color
                                 hovertemplate=hover_template), row=row, col=col)
        if show_u:
            fig.add_trace(go.Scatter(x=list(range(t_max)), y=neu.hist_u, mode='lines', name='u',
                                     line=dict(dash='dot', color='red'), hovertemplate=hover_template), row=row, col=col)
        fig.add_trace(go.Scatter(x=list(range(t_max)), y=sq_wave, mode='lines', name='SqWave',
                                 line=dict(dash='dot', color='red'), hovertemplate=hover_template), row=row, col=col)
        fig.add_trace(go.Scatter(x=list(range(t_max)), y=go_wave / 5, mode='lines', name='GoWave',
                                 line=dict(dash='dot', color='red'), hovertemplate=hover_template), row=row, col=col)

    fig.update_layout(height=300 * n_rows, width=900, title_text="Neuron Dynamics", showlegend=False)
    fig.show()