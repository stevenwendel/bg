import numpy as np
import pandas as pd
from src.neuron import *
from src.utils import *
from src.constants import * 
from src.network import *
from src.validation import *

diagnostic = True

def main():
    periods, input_waves, alphaArray = create_experiment(
                      tMax=tMax,
                      bin_size=100
                      )

    all_neurons = create_neurons(neuron_names)

    # Load DNA into matrix
    dna = load_dna(neuron_names, active_synapses, dna_0)
    
    if diagnostic:
        print("Currently loaded matrix ---")
        display_matrix(dna, neuron_names)

    hello



    return 



if __name__ == "__main__":
    main()

# run network for exp and cont 
# make deepcopies of the neurons after run
# for each neuron, bin all spike trains(n=20) and get difference between exp and cont time-series  
# generate the criteria time-series
# compare with bd = binned_differences()
# score 

