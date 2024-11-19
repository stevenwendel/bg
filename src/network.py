
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


def run_network(neurons, weight_matrix, alpha_array):
    w=weight_matrix
    spikers = []

    # Running the network
    for t in range(1, TMAX):
        # Distributing alphas
        if spikers:
            alpha = alpha_fit(alpha_array, t - 1, TMAX)
            for i, post in enumerate(neurons):
                post.input += alpha * np.dot(spikers, w)[i]

        # Updating V and u, and collecting spikes
        spikers = []
        for neu in neurons:
            neu.hist_V[t], neu.hist_u[t], neu.spike_times[t - 1] = neu.update(I_ext=neu.input[t], sigma=0)
            spikers.append(neu.spiked)
            if neu.spiked:
                neu.hist_V[t - 1] = neu.vpeak