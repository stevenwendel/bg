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

    ga_set = 'small'
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

    # === Defining Criteria === 
    criteria_dict = define_criteria()
    max_score = TMAX // BIN_SIZE * len(CRITERIA_NAMES)

    # === Evaluating DNA ===
    curr_population = [create_dna(GA_CONFIG[ga_set]['DNA_BOUNDS']) for _ in range(GA_CONFIG[ga_set]['POP_SIZE'])]

    save_dict = {}

    for generation in range(GA_CONFIG[ga_set]['NUM_GENERATIONS']):
        print(f"Generation {generation}")
        population_results = []
        save_dict[f'generation{generation}'] = {}

        with Pool(1) as pool:
            args_list = [(dna, all_neurons, alpha_array, input_waves, criteria_dict, generation, max_score) 
                        for dna in curr_population]
            drone_results = pool.imap_unordered(drone_evaluate_dna, args_list)
            for curr_dna, total_score in drone_results:
                population_results.append(
                    {'dna': curr_dna, 
                     'dna_score' : total_score
                     })

        # Quick save to temp dict (repository for all dna across all generations)   
        save_dict[f'generation{generation}'] = population_results

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

