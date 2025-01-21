import sys, os

# Add the src directory to sys.path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.append(src_path)

import numpy as np
import pandas as pd
import shelve
from src.neuron import *
from src.utils import *
from src.constants import * 
from src.network import *
from src.validation import *
from src.viz import *
from src.genetic_algorithm import *
from copy import copy
from datetime import datetime
import time
from multiprocessing import Pool

def main():
    start_time = time.time()
    print(start_time)   

    ga_set = 'xlarge_highMutation'
    ### Settings ###
    os.makedirs('./data', exist_ok=True)
    save_path = f'./data/{ga_set}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl'


    diagnostic = {
        'show_dna_matrix' : False,
        'show_neuron_plots' : False,
        'show_difference_histogram' : False,
        'show_dna_scores': False
    }
    


    
    # === Creating Izhikevich neurons ===
    # Note: these do NOT have instrisic histories; these are generated ONCE at beginning, and copied thereafter
    all_neurons = create_neurons()
    
    # === Preparing Network === 
    splits, input_waves, alpha_array = create_experiment()
    num_periods = len(splits) - 1

    # === Defining Criteria === 
    criteria_dict = define_criteria(num_periods)
    max_score = (num_periods) * len(CRITERIA_NAMES)

    # === Evaluating DNA ===
    curr_population = [create_dna(GA_CONFIG[ga_set]['DNA_BOUNDS']) for _ in range(GA_CONFIG[ga_set]['POP_SIZE'])]

    save_dict = {}

    for generation in range(GA_CONFIG[ga_set]['NUM_GENERATIONS']):
        print(f"Generation {generation}")
        population_results = []
        save_dict[f'generation{generation}'] = {}

        for i, curr_dna in enumerate(curr_population):
            # Loading dna into matrix
            dna_matrix = load_dna(curr_dna) 

            # Running network to score dna
            dna_scores, neuron_data = evaluate_dna(
                dna_matrix=dna_matrix,
                neurons=all_neurons,
                alpha_array=alpha_array,
                input_waves=input_waves,
                criteria=criteria_dict,
                curr_dna=curr_dna
                )
            
            total_score = sum(dna_scores.values())
            print(f'{generation}.{i} === DNA: {curr_dna}') 
            print(f'    === Control: {dna_scores["control"]}/{max_score}')
            print(f'    === Experimental: {dna_scores["experimental"]}/{max_score}')
            print(f'    === Overall: {total_score}({total_score/(2*max_score):.2%})')
            print('\n')
            # Appending results to population_results for spawning next generation (requires list of dicts)
            population_results.append({
                'dna': curr_dna,
                'dna_score' : total_score
            })

            # Adding to master dictionary for quickly aving results to pickle file
            save_dict[f'generation{generation}'][f'iteration{i}'] = {
                'dna': curr_dna,
                'dna_score' : total_score,    
                # 'neuron_data' : neuron_data,
                # 'binned_differences' : binned_differences
            }

            # if generation == 1 and i==8:
            #     diagnostic = {
            #         'show_dna_matrix' : False,
            #         'show_neuron_plots' : True,
            #         'show_difference_histogram' : True,
            #         'show_dna_scores': False
            #     }


            # Show diagnostic feedback
            if diagnostic['show_dna_matrix']:
                print("Currently loaded matrix ---")
                display_matrix(dna_matrix, NEURON_NAMES)

            if diagnostic['show_dna_scores']:
                print(f'{dna_scores=}: {curr_dna}')
            
            if diagnostic['show_neuron_plots']:
                for condition in ['experimental', 'control']:
                    plot_neurons_interactive(neurons=neuron_data[condition], sq_wave=input_waves[0], go_wave=input_waves[1], show_u=False)
                    
            # if diagnostic['show_difference_histogram']:
            #     plot_binned_differences(binned_differences)

        curr_population = spawn_next_population(population_results, GA_CONFIG[ga_set])

    # Pickle run data 
    with open(save_path,'ab') as f:
        pickle.dump(save_dict, f)
    
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    # test_dna=create_dna() # --> use simulated annealing



if __name__ == "__main__":
    main()

# run network for exp and cont 
# make deepcopies of the neurons after run
# for each neuron, bin all spike trains(n=20) and get difference between exp and cont time-series  
# generate the criteria time-series
# compare with bd = binned_differences()
# score 

