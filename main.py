import sys, os

# Add the src directory to sys.path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.append(src_path)

import numpy as np
import pandas as pd

from src.neuron import *
from src.utils import *
from src.constants import * 
from src.network import *
from src.validation import *
from src.dna import *
from src.viz import *
from copy import deepcopy

def main():
    diagnostic = {
        'show_dna_matrix' : True,
        'show_neuron_plots' : False,
        'show_difference_histogram' : False,
        'show_dna_scores': True
    }

    # === Creating Izhikevich neurons ===
    # Note: these do NOT have instrisic histories
    all_neurons = create_neurons()
    
    # === Preparing Network === 
    periods, input_waves, alpha_array = create_experiment()

    # === Defining Criteria === # NEED TO FIX IN VALIDATION.PY FILE
    difference_criteria = define_criteria(len(periods)-1) # Fix this to remove need for epochs; make create_criterion(neuron, on, off)

    # === Evaluating DNA ===
    # Initializing first dna sample: Can be loaded (dna_0) or attained through SA
    test_dna=DNA_0

    # Loop
    for trial in range(MAX_TRIALS):
        pass
    
    # Loading dna into matrix
    dna_matrix = load_dna(test_dna) 

    # Running network to score dna
    dna_score, neuron_data, binned_differences = evaluate_dna(
        dna_matrix=dna_matrix,
        neurons=all_neurons,
        alpha_array=alpha_array,
        input_waves=input_waves,
        criteria=difference_criteria
        )
    
    # Pickle run data 
    with open('./data/run_data.pkl','ab') as f:
        pickle.dump((test_dna, dna_score, neuron_data, binned_differences),f)



    # Show diagnostic feedback
    if diagnostic['show_dna_matrix']:
        print("Currently loaded matrix ---")
        display_matrix(dna_matrix, NEURON_NAMES)

    if diagnostic['show_dna_scores']:
        print(f'{dna_score=}: {test_dna}')
    
    if diagnostic['show_neuron_plots']:
        for condition in ['experimental', 'control']:
            plot_neurons_interactive(
                        neurons=neuron_data[condition],
                        sq_wave=input_waves[0], 
                        go_wave=input_waves[1], 
                        show_u=False)
            
    if diagnostic['show_difference_histogram']:
        plot_binned_differences(binned_differences)

    
    
    # test_dna=create_dna() # --> use simulated annealing



if __name__ == "__main__":
    main()

# run network for exp and cont 
# make deepcopies of the neurons after run
# for each neuron, bin all spike trains(n=20) and get difference between exp and cont time-series  
# generate the criteria time-series
# compare with bd = binned_differences()
# score 

