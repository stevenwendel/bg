import numpy as np
import pandas as pd
from src.utils import *
from src.neuron import *
from src.constants import * 
from src.network import *
from src.validation import *
from copy import deepcopy

# === Running Network and Storing Results ====    
def evaluate_dna(dna):
    # Load DNA into matrix
    weights_0 = load_dna(neuron_names, active_synapses, dna)
    
    if diagnostic:
        print("Currently loaded matrix ---")
        display_matrix(weights_0, neuron_names)
    
    neuron_data = {}

    for condition in ['experimental', 'control']:    
        
        if use_saved:
            with open(f'./data/{condition}_neurons.pkl', 'rb') as f:
                neuron_data[condition]= pickle.load(f)

        else:
            run_network(weight_matrix=weights_0, 
                        neurons=all_neurons, 
                        sq_wave=input_waves[0],
                        go_wave=input_waves[1],
                        t_max=tMax, # I have this written as t_max and tMax throughout; should homogenize
                        dt=dt, 
                        alpha_array=alpha_array,
                        control=False if condition == 'experimental' else True
                        )
            neuron_data[condition] = deepcopy(all_neurons)

        if diagnostic:
            plot_neurons_interactive(neurons=neuron_data[condition],
                                    sq_wave=input_waves[0], 
                                    go_wave=input_waves[1], 
                                    t_max=tMax, 
                                    show_u=False)
    
    # === Getting differences across bins ====        
    binned_differences = get_binned_differences(
        experimental_neurons=neuron_data['experimental'],
        control_neurons=neuron_data['control'],
        bin_size=bin_size)

    if diagnostic:
        plot_binned_differences(
            binned_differences=binned_differences,
            bin_size=bin_size,
            neuron_names=neuron_names
        )

    target_binned_differences=get_neurons(binned_differences,criteria_names)

    # === Defining Criteria ===
    difference_criteria = define_criteria(len(periods)-1) # Fix this to remove need for epochs; make create_criterion(neuron, on, off)
    # === Scoring ===
    score = score_run(target_binned_differences, difference_criteria)
    
    # === Save score to pkl ===
    with open('./data/run_data.pkl','ab') as f:
        pickle.dump((
            dna,
            score,
            neuron_data, 
            binned_differences
            ),f
        )
    return score