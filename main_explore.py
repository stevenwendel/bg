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
    start_time = time.time()
    print(f"Initial memory usage: {get_memory_usage():.2f} MB")

    ### Settings ###
    ga_set = "explore_B"
    os.makedirs('./data', exist_ok=True)
    save_path = f'./data/{ga_set}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl'

    # === Creating Izhikevich neurons ===
    all_neurons = create_neurons()
    
    # === Preparing Network === 
    splits, input_waves, alpha_array = create_experiment()

    # === Defining Criteria === 
    criteria_dict = define_criteria()
    max_score = TMAX // BIN_SIZE * len(CRITERIA_NAMES)
    
    # Run the genetic algorithm
    final_population = run_genetic_algorithm(
        ga_config=GA_CONFIG[ga_set],
        neurons=all_neurons,
        ga_set=ga_set
    )
    
    # Save results
    save_dict = {
        'metadata': {
            'ga_set': ga_set,
            'config': GA_CONFIG[ga_set],
            'start_time': start_time,
            'end_time': time.time()
        },
        'final_population': final_population
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)
    
    print(f"\nRun completed and saved to {save_path}")
    print(f"Final memory usage: {get_memory_usage():.2f} MB")
    print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")
