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




def clean_df(initial_df, threshold=730):
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

def get_unique_representatives_efficient(df, max_synapses=20, call_number=1):
    """
    Returns a DataFrame with unique network topologies (binary connection patterns),
    keeping only configurations with max_synapses or fewer connections.
    For each unique topology, keeps the highest scoring representative.
    """
    print(f"\n=== Call {call_number} ===")
    print(f"Starting with {len(df)} configurations")
    
    # Convert DNA lists to numpy arrays for faster processing
    dna_arrays = np.array([np.array(dna) for dna in df['dna']])
    
    # Create binary masks (1 where weight exists, 0 where no connection)
    binary_patterns = (dna_arrays != 0).astype(int)
    
    # Count synapses in each configuration
    synapse_counts = np.sum(binary_patterns, axis=1)
    
    # Print detailed synapse count information
    print("\nInitial synapse count analysis:")
    for count in range(15, 21):
        num_with_count = np.sum(synapse_counts == count)
        if num_with_count > 0:
            print(f"  {count} synapses: {num_with_count} topologies")
            # Print example DNA for each count
            example_idx = np.where(synapse_counts == count)[0][0]
            example_dna = df.iloc[example_idx]['dna']
            print(f"    Example DNA: {example_dna}")
            print(f"    Non-zero weights: {[w for w in example_dna if w != 0]}")
            # Verify count calculation
            manual_count = sum(1 for w in example_dna if w != 0)
            print(f"    Manual count: {manual_count}")
            print(f"    Numpy count: {synapse_counts[example_idx]}")
    
    # Filter by number of synapses first
    mask = synapse_counts <= max_synapses
    filtered_df = df[mask].copy()
    filtered_patterns = binary_patterns[mask]
    filtered_counts = synapse_counts[mask]  # Keep the correct counts for filtered data
    
    print(f"\nAfter synapse count filter: {len(filtered_df)} configurations")
    
    # Convert binary patterns to tuples for hashing
    pattern_tuples = [tuple(pattern) for pattern in filtered_patterns]
    
    # Create dictionary to store highest scoring example of each pattern
    unique_patterns = {}
    for idx, pattern in enumerate(pattern_tuples):
        score = filtered_df.iloc[idx]['dna_score']
        if pattern not in unique_patterns or score > unique_patterns[pattern]['score']:
            unique_patterns[pattern] = {
                'index': filtered_df.index[idx],
                'score': score,
                'synapse_count': int(filtered_counts[idx]),  # Use the filtered counts
                'pattern': pattern
            }
    
    # Get the indices of the highest scoring representative of each pattern
    unique_indices = [info['index'] for info in unique_patterns.values()]
    
    # Create final DataFrame with unique representatives
    result_df = df.loc[unique_indices].sort_values('dna_score', ascending=False)
    
    # Print final distribution with validation
    print("\nFinal synapse count distribution with validation:")
    final_synapse_counts = np.array([np.sum(np.array(dna) != 0) for dna in result_df['dna']])
    stored_synapse_counts = [info['synapse_count'] for info in unique_patterns.values()]
    
    for count in range(15, 21):
        num_with_count = np.sum(final_synapse_counts == count)
        if num_with_count > 0:
            print(f"\n  {count} synapses: {num_with_count} topologies")
            # Find an example
            example_idx = np.where(final_synapse_counts == count)[0][0]
            example_dna = result_df.iloc[example_idx]['dna']
            stored_count = stored_synapse_counts[example_idx]
            
            print(f"    Example DNA: {example_dna}")
            print(f"    Non-zero weights: {[w for w in example_dna if w != 0]}")
            print(f"    Manual count: {sum(1 for w in example_dna if w != 0)}")
            print(f"    Numpy count: {final_synapse_counts[example_idx]}")
            print(f"    Stored count: {stored_count}")
            
            if stored_count != final_synapse_counts[example_idx]:
                print(f"    WARNING: Count mismatch! Stored: {stored_count}, Calculated: {final_synapse_counts[example_idx]}")
    
    return result_df


def process_parent_row(row, min_score):
    index, parent_dna, parent_score, pass_number = row['index'], row['dna'], row['dna_score'], row['pass_number']
    dna_children = 0
    exiting_dna = []
    
    print(f'Processing: {pass_number}:{index}')

    # Validate parent score first
    if parent_score < min_score:
        print(f"Warning: Parent DNA has score {parent_score} below threshold {min_score}")
        return exiting_dna

    for i in range(len(parent_dna)):
        if parent_dna[i] != 0:
            child_dna = parent_dna.copy()
            child_dna[i] = 0

            # Get detailed score breakdown
            dna_matrix = load_dna(child_dna)
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
                curr_dna=child_dna
            )
            child_score = sum(dna_score.values())
            
            # Print detailed score information
            print(f'Child #{dna_children} ==== Gene removed: {i} | Pass: {pass_number} | Index: {index}')
            print(f'    Parent score: {parent_score}')
            print(f'    Child score: {child_score} (Control: {dna_score["control"]}/{max_score}, Experimental: {dna_score["experimental"]}/{max_score})')
            print(f'    Delta: {child_score-parent_score}')
            
            # Strict score validation
            if child_score >= min_score:
                dna_children += 1
                print(f'    ACCEPTED: Score meets threshold {min_score}')
                exiting_dna.append({
                    'dna': child_dna, 
                    'score': child_score, 
                    'file': row['file'], 
                    'is_child': True,
                    'parent_score': parent_score,
                    'control_score': dna_score['control'],
                    'experimental_score': dna_score['experimental']
                })
            else:
                print(f'    REJECTED: Score below threshold {min_score}')

    # Only add parent to fully_reduced if it has no valid children
    if dna_children == 0 and parent_score >= min_score:
        print(f'Parent added to fully_reduced: No valid children found')
        exiting_dna.append({
            'dna': parent_dna, 
            'score': parent_score, 
            'file': row['file'], 
            'is_child': False,
            'parent_score': parent_score
        })

    return exiting_dna

if __name__ == '__main__':
    mp.freeze_support()
    
    # Start overall timing
    start_time = time.time()
    
    # # Load the data from pass 5
    # with open('/Users/stevenwendel/Documents/GitHub/bg/pass_data_5.pkl', 'rb') as file:
    #     results = pickle.load(file)

    # # Get the DataFrames from the loaded data
    # fully_reduced_dnas = results['fully_reduced_dnas']
    # parent_dnas = results['parent_dnas']

    # # Define the dna_score threshold - use a single consistent threshold
    # min_score = 730

    # # Calculate optimal number of processes (leave one core free for system)
    # num_processes = max(1, mp.cpu_count() - 2)

    # # Start from pass 6 (since we're continuing from pass 5)
    # pass_number = 5

    # print(f"CONTINUING FROM PASS 5")
    # print(f"#### safe-zone parents: {len(fully_reduced_dnas)=}")
    # print(f"#### current parents: {len(parent_dnas)=}")


    # Load the combined data from the .pkl file
    with open('/Users/stevenwendel/Documents/GitHub/bg/combined_data.pkl', 'rb') as file:
        ga_results = pickle.load(file)

    # Create a DataFrame from the combined data
    df = pd.DataFrame(ga_results)
    min_score = 741

    # Initial filtering with strict threshold
    filtered_df = df[df['dna_score'] >= min_score].sort_values(by='dna_score', ascending=False).reset_index(drop=True)
    print(f"Filtered DataFrame: {len(filtered_df)} entries with dna_score >= {min_score}")

    cleaned_df = clean_df(filtered_df, threshold=min_score)
    print(f"Cleaned DataFrame: {len(cleaned_df)} entries after cleaning")

    unique_df = get_unique_representatives_efficient(cleaned_df, max_synapses=20, call_number=1)
    print(f"Unique representatives DataFrame: {len(unique_df)} unique network topologies found")

    # Verify all scores in unique_df meet threshold
    if not all(unique_df['dna_score'] >= min_score):
        print("Warning: Found entries in unique_df with scores below threshold")
        print(unique_df[unique_df['dna_score'] < min_score])

    parent_dnas = get_unique_representatives_efficient(unique_df, max_synapses=20, call_number=2)
    print(f"Unique representatives DataFrame: {len(parent_dnas)} unique network topologies found")

    fully_reduced_dnas = pd.DataFrame(columns=parent_dnas.columns)

    # Calculate optimal number of processes (leave one core free for system)
    num_processes = max(1, mp.cpu_count() - 1)

    pass_number = 0

    while parent_dnas.shape[0] > 0:
        pass_number += 1 
        # Start timing this pass
        pass_start_time = time.time()
        
        # Verify parent scores before processing
        if not all(parent_dnas['dna_score'] >= min_score):
            print("Warning: Found parent_dnas with scores below threshold")
            print(parent_dnas[parent_dnas['dna_score'] < min_score])
            break

        parent_dnas = clean_df(parent_dnas, threshold=min_score)
        print(f"Cleaned DataFrame: {len(parent_dnas)}")

        parent_dnas = get_unique_representatives_efficient(parent_dnas, max_synapses=20, call_number=2)
        print(f"Unique representatives DataFrame: {len(parent_dnas)} unique network topologies found")

        # Create empty df to collect new dnas
        children_dnas = pd.DataFrame(columns=parent_dnas.columns)

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

        # Collect results and validate scores
        for result in results:
            for successor in result:
                # Strict score validation
                if successor['score'] >= min_score:
                    if successor['is_child'] is True:
                        children_dnas.loc[len(children_dnas)] = [successor['dna'], successor['score'], successor['file']]
                    elif successor['is_child'] is False:
                        fully_reduced_dnas.loc[len(fully_reduced_dnas)] = [successor['dna'], successor['score'], successor['file']]
                    else: 
                        print('ERROR')
                else:
                    print(f"Warning: Found DNA with score {successor['score']} below threshold {min_score}")
                    print(f"Parent score was: {successor.get('parent_score', 'unknown')}")
        
        print("Children --> new parents...")
        parent_dnas = children_dnas

        # Verify scores after processing
        if not all(parent_dnas['dna_score'] >= min_score):
            print("Warning: Found parent_dnas with scores below threshold after processing")
            print(parent_dnas[parent_dnas['dna_score'] < min_score])
            break

        # Calculate and print pass timing
        pass_elapsed = time.time() - pass_start_time
        print(f"Pass {pass_number} completed in {pass_elapsed:.2f} seconds")
        print(f"Average time per parent: {pass_elapsed/len(parent_rows):.2f} seconds")

        # Save the current state at the end of the pass
        save_data = {
            'fully_reduced_dnas': fully_reduced_dnas,
            'parent_dnas': parent_dnas,
        }
        with open(f'pass_data_{pass_number}_fix2.pkl', 'wb') as f:
            pickle.dump(save_data, f)
        print(f'{pass_number=}')
        print(f"\nTHE DAY ENDS. THE PARENTS HAVE BEEN CULLED. BUT IT MUST BEGIN... AGAIN!\n")
        print(f"#### safe-zone parents: {len(fully_reduced_dnas)=}")
        print(f"#### current parents: {len(parent_dnas)=}")

    # Calculate and print overall timing
    total_elapsed = time.time() - start_time
    print(f"\nTotal execution time: {total_elapsed:.2f} seconds")
    print(f"Average time per pass: {total_elapsed/(pass_number-5):.2f} seconds") 