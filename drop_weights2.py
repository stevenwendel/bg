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
import time

# Assuming evaluate_dna is a function that takes a DNA sequence and returns its score
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

# Load the DataFrame from the pickle file
df = pd.read_pickle('/Users/stevenwendel/Documents/GitHub/bg/data/unique_df7.pkl')

def improve_dna(dna, original_score, threshold):
    improved_dnas = []
    for i in range(len(dna)):
        if dna[i] != 0:  # Only consider dropping non-zero weights
            new_dna = dna.copy()
            new_dna[i] = 0  # Drop the weight
            new_score = get_dna_score(new_dna)
            if new_score > original_score and new_score >= threshold:
                improved_dnas.append((new_dna, new_score))
    return improved_dnas

def find_improved_dnas(df, threshold):
    all_improved_dnas = []
    
    # Use multiprocessing to speed up the process
    with Pool() as pool:
        results = pool.starmap(improve_dna, [(row['dna'], row['dna_score'], threshold) for index, row in df.iterrows()])
    
    # Flatten the list of improved DNAs
    for improved_dnas in results:
        all_improved_dnas.extend(improved_dnas)
    
    # Convert the list of improved DNAs to a DataFrame
    improved_df = pd.DataFrame(all_improved_dnas, columns=['dna', 'dna_score'])
    return improved_df

# Define a threshold for the minimum acceptable score
score_threshold = 700  # Example threshold

# Find improved DNAs
improved_df = find_improved_dnas(df, score_threshold)

# Save the improved DataFrame to a new pickle file
improved_df.to_pickle('/Users/stevenwendel/Documents/GitHub/bg/data/improved_dna.pkl')