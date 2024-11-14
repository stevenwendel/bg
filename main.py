import numpy as np
import pandas as pd
from src.neuron import *
from src.utils import *
from src.constants import * 
from src.network import *
from src.validation import *

def main(diagnostic = False, use_saved = True):
    
    # === Preparing Network === 
    periods, input_waves, alpha_array = create_experiment(
                      tMax=tMax,
                      bin_size=bin_size
                      )


    # Create list of calibrated Izhikevich neurons
    all_neurons = create_neurons(neuron_names)

    # Load DNA into matrix
    weights_0 = load_dna(neuron_names, active_synapses, dna_0)
    
    if diagnostic:
        print("Currently loaded matrix ---")
        display_matrix(weights_0, neuron_names)


    # === Running Network and Storing Results ====    
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
            neuron_data[condition] = all_neurons

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
    difference_criteria = define_criteria(4) # Fix this to remove need for epochs; make create_criterion(neuron, on, off)
    # === Scoring ===
    score = score_run(target_binned_differences, difference_criteria)

    print(score)
    return 


if __name__ == "__main__":
    main()

# run network for exp and cont 
# make deepcopies of the neurons after run
# for each neuron, bin all spike trains(n=20) and get difference between exp and cont time-series  
# generate the criteria time-series
# compare with bd = binned_differences()
# score 

