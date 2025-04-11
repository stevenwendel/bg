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

    return total_score  # Random score for demonstration

def process_parent_row(row):
    index, parent_dna, parent_score, pass_number = row['index'], row['dna'], row['dna_score'], row['pass_number']
    dna_children = 0
    dna_tuple = tuple(parent_dna)
    
    print(f'Processing: {pass_number}:{index}')

    # Skip if we've seen this configuration before
    if dna_tuple in seen_configurations:
        print(f'Config {index=} previously seen. Skipping.')
        return None

    seen_configurations.add(dna_tuple)

    for i in range(len(parent_dna)):
        if parent_dna[i] != 0:
            child_dna = parent_dna.copy()
            child_dna[i] = 0

            child_score = get_dna_score(child_dna)
            if child_score > min_score:
                dna_children += 1
                print(f'Child #{dna_children} found ==== Index: {index} ==== Score: {child_score} ==== Delta: {child_score-parent_score}')
                return {'dna': child_dna, 'score': child_score, 'file': row['file'], 'is_child': True}  # New child found

    return {'dna': parent_dna, 'score': parent_score, 'file': row['file'], 'is_child': False}

# Load the combined data from the .pkl file
with open('/Users/stevenwendel/Documents/GitHub/bg/combined_data.pkl', 'rb') as file:
    ga_results = pickle.load(file)

# Create a DataFrame from the combined data
df = pd.DataFrame(ga_results)

# Define the dna_score threshold
dna_threshold = 740
# Filter the DataFrame for entries with dna_score above the threshold
filtered_df = df[df['dna_score'] > dna_threshold].sort_values(by='dna_score', ascending=False).reset_index(drop=True)
print(f"Filtered DataFrame: {len(filtered_df)} entries with dna_score > {dna_threshold}")

cleaned_df = clean_df(filtered_df, threshold=dna_threshold)
print(f"Cleaned DataFrame: {len(cleaned_df)} entries after cleaning")

unique_df = get_unique_representatives_efficient(cleaned_df, max_synapses=20)
print(f"Unique representatives DataFrame: {len(unique_df)} unique network topologies found")

# del parent_dnas, children_dnas, seen_configurations, fully_reduced_dnas
parent_dnas = unique_df

fully_reduced_dnas = pd.DataFrame(columns=unique_df.columns)
min_score = 740
seen_configurations = set()
pass_number = 0

while parent_dnas.shape[0] > 0:
    pass_number += 1 
    # Reporting
    print(f'{pass_number=}')
    print(f"A NEW DAY RISES AND THE PARENTS MUST BE CULLED.")
    print(f"#### safe-zone parents: {len(fully_reduced_dnas)=}")
    print(f"#### current parents: {len(parent_dnas)=}")
    print(f"#### seen configurations: {len(seen_configurations)=}")

    ######################################

    # Prepare arguments for multiprocessing
    parent_rows = [{'index': index, 'dna': row['dna'], 'dna_score': row['dna_score'], 'file': row['file'], 'pass_number': pass_number} 
                   for index, row in parent_dnas.iterrows()]
    
    print("Opening pool...")
    with mp.Pool(processes=6) as pool:
        results = pool.map(process_parent_row, parent_rows)

    print("Sorting through results...")
    # Create empty df to collect new dnas
    children_dnas = pd.DataFrame(columns=cleaned_df.columns)

    # Collect results
    for result in results:
        if result['is_child'] is True:
            children_dnas.loc[len(children_dnas)] = [result['dna'], result['score'], result['file']]
        elif result['is_child'] is False:
            fully_reduced_dnas.loc[len(fully_reduced_dnas)] = [result['dna'], result['score'], result['file']]
        else: 
            print('ERROR')
    
    ######################################

    print("Children --> new parents...")
    parent_dnas = children_dnas
    parent_dnas = clean_df(parent_dnas)
    # Save the current state at the end of the pass
    save_data = {
        'fully_reduced_dnas': fully_reduced_dnas,
        'parent_dnas': parent_dnas,
        'seen_configurations': seen_configurations
    }
    with open(f'pass_data_{pass_number}.pkl', 'wb') as f:
        pickle.dump(save_data, f)
