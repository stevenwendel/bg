import numpy as np

from src.constants import *
from src.neuron import *
from src.utils import *

""" Where is my weight matrix and default neurons? I should create a scratch and use that."""
def create_experiment():

    n_bins = TMAX / BIN_SIZE + 1
    periods = np.linspace(0, TMAX, int(n_bins))

    # Creating wave inputs
    sqWave = np.zeros(TMAX)
    goWave = np.zeros(TMAX)
    sqWave[EPOCHS['sample'][0]:EPOCHS['sample'][1]] = CUE_STRENGTH
    goWave[EPOCHS['response'][0]:EPOCHS['response'][0] + GO_DURATION] = GO_STRENGTH
    input_waves = [sqWave, goWave]

    # Create fixed alpha array of length 250
    alphaArray = create_alpha_array(250, L=30)

    return periods, input_waves, alphaArray


# This should be correct. alpha array with t-1 starts at .08, which is the smallest, first value of alpha. 
# This is supposed to be the first input to the neuron at each time step, which drives V.
# Because we manually "spike" the neuron at t-1, t is the first time step that has an non-zeroalpha input.
def run_network(neurons, weight_matrix, alpha_array): 
    spikers = np.zeros(len(neurons))  # Initialize with zeros for all neurons
        
    # Running the network
    for t in range(1, TMAX):
        # Distributing alphas
        if np.any(spikers):
            alpha = alpha_fit(alpha_array, t, TMAX)
            for i, post in enumerate(neurons):
                post.input += alpha * np.dot(spikers, weight_matrix)[i]

        # Collecting spikes from all neurons from previous time step
        spikers = np.zeros(len(neurons))  # Initialize with zeros for all neurons

        for i, neu in enumerate(neurons):
            neu.hist_V[t], neu.hist_u[t], neu.spike_times[t - 1] = neu.update(I_ext=neu.input[t], sigma=0)
            spikers[i] = 1.0 if neu.spiked else 0.0  # Use 1.0 for spikes instead of True
            if neu.spiked:
                neu.hist_V[t - 1] = neu.vpeak
        
        