import pandas as pd
import numpy as np
import pickle
from src.constants import *
from src.neuron import *
from src.utils import *
from src.network import *
from src.validation import *
from src.viz import *
from src.genetic_algorithm import *
from multiprocessing import Pool
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

def get_representatives(df, max_synapses=20):
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
    print(f'    === DNA: {curr_dna}') 
    print(f'    === Control: {dna_score["control"]}/{max_score}')
    print(f'    === Experimental: {dna_score["experimental"]}/{max_score}')
    print(f'    === Overall: {total_score}({total_score/(2*max_score):.2%})')
    print('\n')

    return total_score  # Random score for demonstration


# Load and prep data
pkl_file = '/Users/stevenwendel/Documents/GitHub/bg/data/J_high_gen_2025-03-13_04-00-56.pkl'

with open(pkl_file, 'rb') as f: 
    ga_results = pickle.load(f)

initial_df = flatten_pkl(ga_results)
cleaned_df = clean_df(initial_df, threshold=728)
parent_dnas = get_representatives(cleaned_df, max_synapses=20).reset_index(drop=True)

def process_parent(row, seen_configurations, permissible_score_change):
    """Process a single parent DNA and return its children and status"""
    children = []
    parent_dna = row['dna']
    parent_score = row['dna_score']
    dna_children = 0
    
    dna_tuple = tuple(parent_dna)
    # Skip if we've seen this configuration before
    if dna_tuple in seen_configurations:
        return [], False
    
    for i in range(len(parent_dna)):
        if parent_dna[i] != 0:
            child_dna = parent_dna.copy()
            child_dna[i] = 0
            child_score = get_dna_score(child_dna)
            
            if child_score - parent_score > permissible_score_change:
                children.append({
                    'generation': row['generation'],
                    'dna': child_dna,
                    'dna_score': child_score
                })
                dna_children += 1
                print(f'New child! | Score Change: {child_score - parent_score} | Num Children: {dna_children}')
    
    return children, dna_children == 0

def main():
    # Your existing setup code here
    pkl_file = '/Users/stevenwendel/Documents/GitHub/bg/data/J_high_gen_2025-03-13_04-00-56.pkl'

    with open(pkl_file, 'rb') as f: 
        ga_results = pickle.load(f)

    initial_df = flatten_pkl(ga_results)
    cleaned_df = clean_df(initial_df, threshold=728)
    parent_dnas = get_representatives(cleaned_df, max_synapses=20).reset_index(drop=True)

    # Begin culling with multiprocessing
    fully_reduced_dnas = pd.DataFrame(columns=cleaned_df.columns)
    permissible_score_change = -2
    seen_configurations = set()

    with Pool() as pool:
        while parent_dnas.shape[0] > 0:
            print(f"A new day rises and the parents must be culled. The parents who have made it to the safe zone are:")
            display(fully_reduced_dnas)
            print("The current parents:")
            display(parent_dnas)
            
            # Process all parents in parallel
            process_func = partial(process_parent, 
                                 seen_configurations=seen_configurations.copy(), 
                                 permissible_score_change=permissible_score_change)
            results = pool.map(process_func, [row for _, row in parent_dnas.iterrows()])
            
            # Collect all children and fully reduced parents
            all_children = []
            newly_reduced = []
            
            for (children, is_reduced), (_, parent_row) in zip(results, parent_dnas.iterrows()):
                seen_configurations.add(tuple(parent_row['dna']))
                all_children.extend(children)
                if is_reduced:
                    newly_reduced.append(parent_row)
            
            # Create new children DataFrame
            children_dnas = pd.DataFrame(all_children)
            
            # Add newly reduced parents to fully_reduced_dnas
            if newly_reduced:
                fully_reduced_dnas = pd.concat([fully_reduced_dnas, pd.DataFrame(newly_reduced)], 
                                             ignore_index=True)
            
            # Update parent_dnas for next iteration
            parent_dnas = children_dnas if not children_dnas.empty else pd.DataFrame(columns=cleaned_df.columns)

    print("Final reduced configurations:")
    display(fully_reduced_dnas)

if __name__ == '__main__':
    main()