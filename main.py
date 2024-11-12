import numpy as np
import pandas as pd
from src.neuron import *
from src.utils import *
from src.constants import * 
from src.network import *
from src.validation import *

def main():
    # Timesetting
    tMax = 5000
    dt = 1

    go_epoch_length = 400
    go_signal_duration = 100

    # Creating epochs (unnecessary artifact)
    epochs = {
    'sample'   : [1000, 2000], #should there be a [0,1000] epoch?
    'delay'    : [2000, 3000],
    'response' : [3000, 4000] #should this be up to 5000?
    }

    bin_size = 20
    n_bins = tMax / bin_size

    periods = np.linspace(0, tMax, n_bins)


    # tMax = sum(end - start for start, end in epochs.values())

    # Creating wave inputs
    sqWave = np.zeros(tMax)
    goWave = np.zeros(tMax)

    sqWave[epochs['sample'][0]:epochs['sample'][1]] = 145.
    goWave[epochs['response'][0]:epochs['response'][0] + go_signal_duration] = 850.

    # Instantiating neurons
    neurons = []
    for neu in nodes:
        if neu in ["MSN1", "MSN2", "MSN3"]:
            neuron_instance = Izhikevich(name = neu, neuron_type="msn")
        else:
            neuron_instance = Izhikevich(name = neu, neuron_type="rs")
        globals()[neu] = neuron_instance # Makes instances available globally
        neurons.append(neuron_instance) # Creates a list of all Iz neurons; note, these are the actual objects, not a list of names!

    SNR1.E = SNR2.E = SNR3.E = 112. 
    PPN.E = 100.                    

    # Create fixed alpha array of length 250
    alphaArray = createAlphaArray(250)

    # Load DNA
    dna = load_dna(my_free_weights, dna_0)

    return 



if __name__ == "__main__":
    main()

# run network for exp and cont 
# make deepcopies of the neurons after run
# for each neuron, bin all spike trains(n=20) and get difference between exp and cont time-series  
# generate the criteria time-series
# compare with bd = binned_differences()
# score 

