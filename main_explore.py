# imports  
import sys, os

# # Add the src directory to sys.path
# src_path = os.path.join(os.path.dirname(__file__), 'src')
# sys.path.append(src_path)

import numpy as np
import pandas as pd
import shelve
import importlib
from src.neuron import *
from src.utils import *
from src.constants import * 
from src.network import *
from src.validation import *
from src.viz import *
from src.genetic_algorithm import *
from copy import copy
from datetime import datetime
from IPython.display import display, HTML
import sys, os
import time
from multiprocessing import Pool



### Settings ###
ga_set = "explore_B"
os.makedirs('./data', exist_ok=True)
save_path = f'./data/{ga_set}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl'

all_neurons = create_neurons()
splits, input_waves, alpha_array = create_experiment()
criteria_dict = define_criteria()
max_score = TMAX // BIN_SIZE * len(CRITERIA_NAMES)

# initial_dna = create_dna_string(new_jh_weights, ACTIVE_SYNAPSES)
initial_dna = [497, 0, 0, 1000, -108, -14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -459, 0, 0, 88, 0, 0, 0, 0, 628, 0, 0, 0, 0, 0, 0, 0, 353, 148, 0, 0, 0, 0, 0, 0, 0, 0, -691, 0, 0, 0, 28, 275, 179]

sigma= GA_CONFIG[ga_set]['MUT_SIGMA'] * 10

if __name__ == "__main__":
    curr_population = []
    for _ in range(GA_CONFIG[ga_set]['POP_SIZE']):
        randomized_dna = np.round(np.random.normal(loc=initial_dna, scale=sigma)).astype(int)
        curr_population.append(randomized_dna)

    save_dict = {}

    for generation in range(GA_CONFIG[ga_set]['NUM_GENERATIONS']):
        print(f"Generation {generation}")
        population_results = []
        save_dict[f'{generation}'] = {}

        with Pool() as pool:
            args_list = [(dna, all_neurons, alpha_array, input_waves, criteria_dict, generation, max_score) 
                        for dna in curr_population]
            drone_results = pool.imap_unordered(drone_evaluate_dna, args_list)
            for curr_dna, total_score in drone_results:
                population_results.append(
                    {'dna': curr_dna, 
                        'dna_score' : total_score
                        })

        # Quick save to temp dict (repository for all dna across all generations)   
        save_dict[f'{generation}'] = population_results

        curr_population = spawn_next_population(population_results, GA_CONFIG[ga_set], generation)

    # Pickle run data 
    with open(save_path,'ab') as f:
        pickle.dump(save_dict, f)
