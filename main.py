import sys, os
import gc
import pickle
import matplotlib.pyplot as plt

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
    print(f"Initial memory usage: {get_memory_usage():.2f} MB")

    ga_set = "large"
    ### Settings ###
    os.makedirs('./data', exist_ok=True)
    save_path = f'./data/{ga_set}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl'

    # Performance monitoring settings
    PERFORMANCE_THRESHOLD = 0.70  # Minimum score ratio (score/max_possible_score) to consider successful
    CHECK_GENERATIONS = 200      # Number of generations to check before considering a restart
    MAX_RESTARTS = 20           # Maximum number of times to restart before giving up

    # Multiprocessing settings
    NUM_PROCESSES = os.cpu_count() - 1  # Leave one CPU free
    CHUNK_SIZE = 1  # Process one DNA at a time for better load balancing

    diagnostic = {
        'show_dna_matrix' : False,
        'show_neuron_plots' : False,
        'show_difference_histogram' : False,
        'show_dna_scores': False
    }
        
    # === Creating Izhikevich neurons ===
    all_neurons = create_neurons()
    
    # === Preparing Network === 
    splits, input_waves, alpha_array = create_experiment()

    # === Defining Criteria === 
    criteria_dict = define_criteria()
    max_score = TMAX // BIN_SIZE * len(CRITERIA_NAMES) * 2
    threshold_score = max_score * PERFORMANCE_THRESHOLD
    print(f'{max_score=} {threshold_score=}')
    restart_count = 0
    while restart_count < MAX_RESTARTS:
        print(f"\nStarting run {restart_count + 1} of {MAX_RESTARTS}")
        
        # Initialize save_dict with metadata
        save_dict = {
            'metadata': {
                'ga_set': ga_set,
                'config': GA_CONFIG[ga_set],
                'start_time': start_time,
                'generation': 0,
                'restart_count': restart_count,
                'num_processes': NUM_PROCESSES
            },
            'best_dna': None,
            'best_score': float('-inf')
        }

        # === Evaluating DNA ===
        curr_population = [create_dna(GA_CONFIG[ga_set]['DNA_BOUNDS']) for _ in range(GA_CONFIG[ga_set]['POP_SIZE'])]

        for generation in range(GA_CONFIG[ga_set]['NUM_GENERATIONS']):
            print(f"=== Generation {generation} ===")
            population_results = []
            
            # Create a process pool with proper error handling
            try:
                with Pool(processes=NUM_PROCESSES) as pool:
                    args_list = [(dna, all_neurons, alpha_array, input_waves, criteria_dict, generation, max_score) 
                                for dna in curr_population]
                    
                    # Use imap with chunk_size for better load balancing
                    for curr_dna, total_score in pool.imap_unordered(
                        drone_evaluate_dna, 
                        args_list,
                        chunksize=CHUNK_SIZE
                    ):
                        population_results.append({
                            'dna': curr_dna, 
                            'dna_score': total_score
                        })
                        
                        # Update best DNA if better score found
                        if total_score > save_dict['best_score']:
                            save_dict['best_score'] = total_score
                            save_dict['best_dna'] = curr_dna
                            save_dict['metadata']['generation'] = generation
            except Exception as e:
                print(f"Error in process pool: {e}")
                # Clean up any remaining processes
                pool.terminate()
                pool.join()
                raise

            # Quick save to temp dict (repository for all dna across all generations)   
            save_dict[f'gen_{generation}'] = {
                'population': population_results,
                'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            }

            curr_population, mutation_stats = spawn_next_population(population_results, GA_CONFIG[ga_set], generation)

            # Check if we've reached the threshold
            if generation == CHECK_GENERATIONS:
                if save_dict['best_score'] >= threshold_score:
                    print(f"\nSuccess! Reached threshold score of {threshold_score} at generation {generation}")
                    print(f"Best score achieved: {save_dict['best_score']}")
                    print("Continuing to completion...")
                else:
                    print(f"\nFailed to reach threshold score of {threshold_score} after {CHECK_GENERATIONS} generations")
                    print(f"Best score achieved: {save_dict['best_score']}")
                    print(f"Restarting run {restart_count + 1} of {MAX_RESTARTS}")
                    break  # Break out of the generation loop to trigger restart

            # Pickle run data 
            try:
                with open(save_path, 'wb') as f:
                    pickle.dump(save_dict, f)
                # print(f"(Memory usage after generation {generation}: {get_memory_usage():.2f} MB)")
                
                # Calculate and print population statistics
                scores = [p['dna_score'] for p in population_results]
                max_score = max(scores)
                avg_score = sum(scores) / len(scores)
                diversity = calculate_population_diversity(population_results, GA_CONFIG[ga_set]['DNA_BOUNDS'])
                
                print(f"  Max Score (Avg): {max_score:.2f} ({avg_score:.2f}) === Population Diversity: {diversity:.3f} === Sigma:{mutation_stats['mutation_sigma']:.3f}")
                
                # Force garbage collection
                gc.collect()
            except IOError as e:
                print(f"Error saving data: {e}")
                # Optionally save to a backup file
                backup_path = f'./data/backup_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl'
                try:
                    with open(backup_path, 'wb') as f:
                        pickle.dump(save_dict, f)
                    print(f"Data saved to backup file: {backup_path}")
                except IOError as e:
                    print(f"Failed to save backup: {e}")

        # If we've completed all generations without reaching threshold, restart
        if save_dict['best_score'] < threshold_score:
            save_path = f'./data/{ga_set}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl'
            restart_count += 1
            if restart_count < MAX_RESTARTS:
                print(f"\nFailed to reach threshold score of {threshold_score}")
                print(f"Best score achieved: {save_dict['best_score']}")
                print(f"Restarting run {restart_count + 1} of {MAX_RESTARTS}")
            else:
                print(f"\nFailed to reach threshold after {MAX_RESTARTS} attempts")
                print(f"Best score achieved: {save_dict['best_score']}")
        else:
            # Only break the restart loop if we've completed all generations
            if generation == GA_CONFIG[ga_set]['NUM_GENERATIONS'] - 1:
                break
    
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

