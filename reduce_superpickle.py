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
from multiprocessing import Manager

# Load the combined data from the .pkl file
with open('/Users/stevenwendel/Documents/GitHub/bg/combined_data.pkl', 'rb') as file:
    ga_results = pickle.load(file)

#unused?

# Create a DataFrame from the combined data
df = pd.DataFrame(ga_results)


# Define the dna_score threshold
dna_threshold = 730
# Filter the DataFrame for entries with dna_score above the threshold
filtered_df = df[df['dna_score'] > dna_threshold].sort_values(by='dna_score', ascending=False).reset_index(drop=True)

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

cleaned_df = clean_df(filtered_df, threshold=728)

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

cleaned_df = get_unique_representatives_efficient(cleaned_df, max_synapses=20)

def get_dna_score(curr_dna):
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
    print(f'    === DNA: {curr_dna}') 
    print(f'    === Control: {dna_score["control"]}/{max_score}')
    print(f'    === Experimental: {dna_score["experimental"]}/{max_score}')
    print(f'    === Overall: {total_score}({total_score/(2*max_score):.2%})')
    print('\n')

    return total_score  # Random score for demonstration
# del parent_dnas, children_dnas, seen_configurations, fully_reduced_dnas
parent_dnas = cleaned_df

fully_reduced_dnas = pd.DataFrame(columns=cleaned_df.columns)
min_score = 728
seen_configurations = set()

def process_parent_dna(args):
    parent_row, min_score = args
    parent_dna = parent_row['dna']
    parent_score = parent_row['dna_score']
    results = []
    
    for i in range(len(parent_dna)):
        if parent_dna[i] != 0:
            child_dna = parent_dna.copy()
            child_dna[i] = 0
            
            child_score = get_dna_score(child_dna)
            if child_score > min_score:
                results.append({
                    'dna': child_dna,
                    'dna_score': child_score,
                    'file': parent_row['file']
                })
    
    return results

while parent_dnas.shape[0] > 0:
    print(f"A new day rises and the parents must be culled. The parents who have made it to the safe zone are:")
    display(fully_reduced_dnas)
    print("The current parents:")
    display(parent_dnas)
    print("seen configurations")
    print(seen_configurations)

    # Create process pool
    num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes)
    
    # Prepare arguments for parallel processing
    args = [(row, min_score) for _, row in parent_dnas.iterrows()]
    
    # Process parents in parallel
    all_results = pool.map(process_parent_dna, args)
    pool.close()
    pool.join()
    
    # Flatten results and create new DataFrame
    children_dnas = pd.DataFrame(columns=cleaned_df.columns)
    for results in all_results:
        for result in results:
            dna_tuple = tuple(result['dna'])
            if dna_tuple not in seen_configurations:
                seen_configurations.add(dna_tuple)
                children_dnas.loc[len(children_dnas)] = result
    
    # Add parents with no children to fully_reduced_dnas
    for _, row in parent_dnas.iterrows():
        dna_tuple = tuple(row['dna'])
        if dna_tuple not in seen_configurations:
            fully_reduced_dnas.loc[len(fully_reduced_dnas)] = row
    
    parent_dnas = children_dnas

# Save fully_reduced_dnas as a pkl file
fully_reduced_dnas.to_pickle('fully_reduced_dnas.pkl')