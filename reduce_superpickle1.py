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
from tqdm import tqdm
import time
import gc
import signal
import psutil

# Memory management
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def check_memory_threshold(threshold_mb=8000):
    """Check if memory usage exceeds threshold"""
    return get_memory_usage() > threshold_mb

# Signal handler for graceful interruption
def signal_handler(signum, frame):
    print("\nReceived interrupt signal. Saving progress and exiting...")
    save_progress()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Load the combined data from the .pkl file
print("Loading data...")
with open('/Users/stevenwendel/Documents/GitHub/bg/combined_data.pkl', 'rb') as file:
    ga_results = pickle.load(file)

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

def save_progress():
    """Save current progress to disk"""
    try:
        print("\nSaving progress...")
        with open('progress.pkl', 'wb') as f:
            pickle.dump({
                'fully_reduced_dnas': fully_reduced_dnas,
                'seen_configurations': list(seen_configurations),
                'parent_dnas': parent_dnas
            }, f)
        print("Progress saved successfully")
    except Exception as e:
        print(f"Error saving progress: {e}")

def load_progress():
    """Load progress from disk if it exists"""
    try:
        if os.path.exists('progress.pkl'):
            print("Loading previous progress...")
            with open('progress.pkl', 'rb') as f:
                progress = pickle.load(f)
                return progress['fully_reduced_dnas'], set(progress['seen_configurations']), progress['parent_dnas']
    except Exception as e:
        print(f"Error loading progress: {e}")
    return None, None, None

# Try to load previous progress
fully_reduced_dnas, seen_configurations, parent_dnas = load_progress()

# If no progress found, initialize
if fully_reduced_dnas is None:
    fully_reduced_dnas = pd.DataFrame(columns=cleaned_df.columns)
    seen_configurations = set()
    parent_dnas = cleaned_df

def process_parent_dna(args):
    try:
        parent_row, min_score, parent_idx, total_parents = args
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
    except Exception as e:
        print(f"Error processing parent DNA: {e}")
        return []

def process_batch(args):
    """Process a batch of parents and return results with progress info"""
    batch_args, batch_idx, total_batches = args
    results = []
    for parent_args in batch_args:
        results.extend(process_parent_dna(parent_args))
    return results

def main():
    global fully_reduced_dnas, seen_configurations, parent_dnas
    
    while parent_dnas.shape[0] > 0:
        print(f"\n{'='*80}")
        print(f"Starting new iteration with {len(parent_dnas)} parents")
        print(f"Current fully reduced DNAs: {len(fully_reduced_dnas)}")
        print(f"Total unique configurations seen: {len(seen_configurations)}")
        print(f"Current memory usage: {get_memory_usage():.2f} MB")
        print(f"{'='*80}\n")
        
        # Limit number of processes to prevent system overload
        num_processes = min(6, mp.cpu_count() - 1)  # Leave one core free
        
        # Create process pool
        with mp.Pool(processes=num_processes) as pool:
            # Prepare arguments for parallel processing
            total_parents = len(parent_dnas)
            args = [(row, min_score, idx, total_parents) for idx, (_, row) in enumerate(parent_dnas.iterrows())]
            
            # Split into smaller batches for better memory management
            batch_size = max(1, len(args) // (num_processes * 4))
            batches = [args[i:i + batch_size] for i in range(0, len(args), batch_size)]
            batch_args = [(batch, i, len(batches)) for i, batch in enumerate(batches)]
            
            # Process parents in parallel with progress bar
            print(f"Processing {len(batches)} batches with {num_processes} processes...")
            all_results = []
            with tqdm(total=len(batches), desc="Processing batches") as pbar:
                for result in pool.imap_unordered(process_batch, batch_args):
                    all_results.extend(result)
                    pbar.update(1)
                    
                    # Check memory usage periodically
                    if check_memory_threshold():
                        print("\nMemory threshold reached. Saving progress and cleaning up...")
                        save_progress()
                        gc.collect()
        
        # Flatten results and create new DataFrame
        print("\nProcessing results...")
        children_dnas = pd.DataFrame(columns=cleaned_df.columns)
        new_configurations = 0
        
        for result in tqdm(all_results, desc="Adding valid children"):
            dna_tuple = tuple(result['dna'])
            if dna_tuple not in seen_configurations:
                seen_configurations.add(dna_tuple)
                children_dnas.loc[len(children_dnas)] = result
                new_configurations += 1
        
        # Add parents with no children to fully_reduced_dnas
        print("\nChecking parents for fully reduced status...")
        new_fully_reduced = 0
        for _, row in tqdm(parent_dnas.iterrows(), total=len(parent_dnas), desc="Checking parents"):
            dna_tuple = tuple(row['dna'])
            if dna_tuple not in seen_configurations:
                fully_reduced_dnas.loc[len(fully_reduced_dnas)] = row
                new_fully_reduced += 1
        
        print(f"\nIteration Summary:")
        print(f"- New configurations found: {new_configurations}")
        print(f"- New fully reduced DNAs: {new_fully_reduced}")
        print(f"- Total configurations seen: {len(seen_configurations)}")
        print(f"- Total fully reduced DNAs: {len(fully_reduced_dnas)}")
        
        parent_dnas = children_dnas
        
        # Save progress after each iteration
        save_progress()
        
        # Clean up memory
        gc.collect()

    # Save final results
    print("\nSaving final results...")
    fully_reduced_dnas.to_pickle('fully_reduced_dnas.pkl')
    print("Done!")

if __name__ == '__main__':
    mp.freeze_support()
    main()