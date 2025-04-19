import sys, os
import pickle
import multiprocessing as mp
import numpy as np
import pandas as pd
from src.neuron import *
from src.utils import *
from src.constants import * 
from src.network import *
from src.validation import *
from src.viz import *
from src.genetic_algorithm import *
from functools import partial
import ast
import time


# Should I use a GUI to show myself the progress? Timing, pass#, processes remaining, ect.

def clean_df(initial_df, threshold=700):
    print(f"Initial rows: {len(initial_df)}")
    
    # Create a copy after filtering to avoid chained indexing
    cleaned_df = initial_df[initial_df['dna_score'] >= threshold].copy()
    print(f"Rows after threshold filter: {len(cleaned_df)}")
    
    # Add tuple column to the copy
    cleaned_df.loc[:, 'dna_tuple'] = cleaned_df['dna'].apply(tuple)
    
    # Drop duplicates and create new copy
    cleaned_df = cleaned_df.drop_duplicates(subset='dna_tuple')
    print(f"Rows after removing duplicates: {len(cleaned_df)}")
    
    # Drop the temporary column and sort
    cleaned_df = cleaned_df.drop('dna_tuple', axis=1)
    cleaned_df = cleaned_df.sort_values(by='dna_score', ascending=False)    
    return cleaned_df

def get_unique_representatives_efficient(df, max_synapses=20):
    """
    Returns a DataFrame with unique network topologies (binary connection patterns),
    keeping only configurations with max_synapses or fewer connections.
    For each unique topology, keeps the highest scoring representative.
    """
    # Convert DNA lists to numpy arrays for faster processing
    dna_arrays = np.array([np.array(dna) for dna in df['dna']])
    
    # Create binary masks (1 where weight exists, 0 where no connection)
    binary_patterns = (dna_arrays != 0).astype(int)
    
    # Count synapses in each configuration
    synapse_counts = np.sum(binary_patterns, axis=1)
    
    # Filter by number of synapses first
    mask = synapse_counts <= max_synapses
    filtered_df = df[mask].copy()
    filtered_patterns = binary_patterns[mask]
    
    # Convert binary patterns to tuples for hashing
    pattern_tuples = [tuple(pattern) for pattern in filtered_patterns]
    
    # Create dictionary to store highest scoring example of each pattern
    unique_patterns = {}
    for idx, pattern in enumerate(pattern_tuples):
        score = filtered_df.iloc[idx]['dna_score']
        if pattern not in unique_patterns or score > unique_patterns[pattern]['score']:
            unique_patterns[pattern] = {
                'index': filtered_df.index[idx],
                'score': score
            }
    
    # Get the indices of the highest scoring representative of each pattern
    unique_indices = [info['index'] for info in unique_patterns.values()]
    
    # Create final DataFrame with unique representatives
    result_df = df.loc[unique_indices].sort_values('dna_score', ascending=False)
    
    print(f"Found {len(result_df)} unique network topologies with {max_synapses} or fewer synapses")
    return result_df

def get_dna_score(curr_dna):
    # Convert tuple back to list if needed
    if isinstance(curr_dna, tuple):
        curr_dna = list(curr_dna)
    
    dna_matrix = load_dna(curr_dna)
    
    # === Preparing Network === 
    all_neurons = create_neurons()
    splits, input_waves, alpha_array = create_experiment()
    criteria_dict = define_criteria()
    max_score = TMAX // BIN_SIZE * len(CRITERIA_NAMES)

    dna_score, neuron_data = evaluate_dna(
        dna_matrix=dna_matrix,
        neurons=all_neurons,
        alpha_array=alpha_array,
        input_waves=input_waves,
        criteria=criteria_dict,
        curr_dna=curr_dna
    )
    total_score = sum(dna_score.values())

    return total_score

def process_parent_row(row, min_score):
    index, parent_dna, parent_score, pass_number = row['index'], row['dna'], row['dna_score'], row['pass_number']
    dna_children = 0
    exiting_dna = []
    
    print(f'Processing: {pass_number}:{index}')

    for i in range(len(parent_dna)):
        if parent_dna[i] != 0:
            child_dna = parent_dna.copy()
            child_dna[i] = 0

            child_score = get_dna_score(child_dna)
            if child_score >= min_score:
                dna_children += 1
                print(f'Child #{dna_children} found ==== Gene removed: {i} | Pass: {pass_number} | Index: {index} | Parent score: {parent_score} | Child Score: {child_score} | Delta: {child_score-parent_score}')
                exiting_dna.append({'dna': child_dna, 'score': child_score, 'file': row['file'], 'is_child': True} ) # New child found

    if dna_children == 0:
        exiting_dna.append({'dna': parent_dna, 'score': parent_score, 'file': row['file'], 'is_child': False})

    return exiting_dna

if __name__ == '__main__':
    mp.freeze_support()
    
    # Start overall timing
    start_time = time.time()
    
    # Load the combined data from the .pkl file
    with open('/Users/stevenwendel/Documents/GitHub/bg/combined_data.pkl', 'rb') as file:
        ga_results = pickle.load(file)

    # Create a DataFrame from the combined data
    df = pd.DataFrame(ga_results)

    # Define the dna_score threshold
    dna_threshold = min_score = 730
    # Filter the DataFrame for entries with dna_score above the threshold
    filtered_df = df[df['dna_score'] > dna_threshold].sort_values(by='dna_score', ascending=False).reset_index(drop=True)
    print(f"Filtered DataFrame: {len(filtered_df)} entries with dna_score > {dna_threshold}")

    cleaned_df = clean_df(filtered_df, threshold=dna_threshold)
    print(f"Cleaned DataFrame: {len(cleaned_df)} entries after cleaning")

    unique_df = get_unique_representatives_efficient(cleaned_df, max_synapses=20)
    print(f"Unique representatives DataFrame: {len(unique_df)} unique network topologies found")

    parent_dnas = unique_df
    fully_reduced_dnas = pd.DataFrame(columns=unique_df.columns)

    # Calculate optimal number of processes (leave one core free for system)
    num_processes = max(1, mp.cpu_count() - 2)

    pass_number = 0

    print(f"A NEW DAY RISES AND THE PARENTS MUST BE CULLED.")
    print(f"#### safe-zone parents: {len(fully_reduced_dnas)=}")
    print(f"#### current parents: {len(parent_dnas)=}")


    while parent_dnas.shape[0] > 0:
        pass_number += 1 
        # Start timing this pass
        pass_start_time = time.time()
        

        parent_dnas = clean_df(parent_dnas, threshold=dna_threshold)
        print(f"Cleaned DataFrame: {len(parent_dnas)}")

        parent_dnas = get_unique_representatives_efficient(parent_dnas, max_synapses=20)
        print(f"Unique representatives DataFrame: {len(unique_df)} unique network topologies found")

        # Reporting

        # Create empty df to collect new dnas
        children_dnas = pd.DataFrame(columns=cleaned_df.columns)

        # Prepare arguments for multiprocessing
        parent_rows = [{'index': index, 
                        'dna': row['dna'], 
                        'dna_score': row['dna_score'], 
                        'file': row['file'], 
                        'pass_number': pass_number, 
                        'min_score': min_score} 
                            for index, row in parent_dnas.iterrows()]
        
        print(f"Opening pool with {num_processes} processes...")
        with mp.Pool(processes=num_processes) as pool:
            process_func = partial(process_parent_row, min_score=min_score)
            results = pool.map(process_func, parent_rows)

        print("Sorting through results...")

        # Collect results
        for result in results:
            for successor in result:
                if successor['is_child'] is True:
                    children_dnas.loc[len(children_dnas)] = [successor['dna'], successor['score'], successor['file']]
                elif successor['is_child'] is False:
                    fully_reduced_dnas.loc[len(fully_reduced_dnas)] = [successor['dna'], successor['score'], successor['file']]
                else: 
                    print('ERROR')
        
        print("Children --> new parents...")
        parent_dnas = children_dnas

        # Calculate and print pass timing
        pass_elapsed = time.time() - pass_start_time
        print(f"Pass {pass_number} completed in {pass_elapsed:.2f} seconds")
        print(f"Average time per parent: {pass_elapsed/len(parent_rows):.2f} seconds")

        # Save the current state at the end of the pass
        save_data = {
            'fully_reduced_dnas': fully_reduced_dnas,
            'parent_dnas': parent_dnas
        }
        with open(f'pass_data_{pass_number}_with_representatives.pkl', 'wb') as f:
            pickle.dump(save_data, f)
        print(f'{pass_number=}')
        print(f"\nTHE DAY ENDS. THE PARENTS HAVE BEEN CULLED. BUT IT MUST BEGIN... AGAIN!\n")
        print(f"#### safe-zone parents: {len(fully_reduced_dnas)=}")
        print(f"#### current parents: {len(parent_dnas)=}")

    # Calculate and print overall timing
    total_elapsed = time.time() - start_time
    print(f"\nTotal execution time: {total_elapsed:.2f} seconds")
    print(f"Average time per pass: {total_elapsed/pass_number:.2f} seconds")
