import pandas as pd
import numpy as np
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

# Assuming evaluate_dna is a function that takes a DNA sequence and returns its score
def get_dna_score(curr_dna):
        
    dna_matrix = load_dna(curr_dna)
    
    # === Preparing Network === 
    all_neurons = create_neurons()
    splits, input_waves, alpha_array = create_experiment()
    criteria_dict = define_criteria()
    max_score = TMAX//BIN_SIZE * len(CRITERIA_NAMES)

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

    return dna_score  # Random score for demonstration

def evaluate_dna_change(original_dna, index, original_score):
    # Create a copy of the original DNA and set the specified index to zero
    modified_dna = original_dna.copy()
    modified_dna[index] = 0
    
    # Evaluate the modified DNA
    new_score_dict = get_dna_score(modified_dna)
    
    # Assuming new_score_dict is a dictionary with keys like 'control' and 'experimental'
    # You need to decide how to combine these into a single score
    new_score = sum(new_score_dict.values())  # Example: sum all scores
    
    # Calculate the change in score
    score_change = original_score - new_score
    return score_change

def process_dna_row(row):
    dna = row['dna'] 
    original_score = row['dna_score']
    
    # Find indices of non-zero elements
    non_zero_indices = [i for i, x in enumerate(dna) if x != 0]
    
    # Use partial to fix the original DNA and score
    evaluate_partial = partial(evaluate_dna_change, dna, original_score=original_score)
    
    # Use multiprocessing to evaluate changes in parallel
    with Pool() as pool:
        score_changes = pool.map(evaluate_partial, non_zero_indices)
    
    # Create a list of score changes with the same length as the original DNA
    score_change_list = [0] * len(dna)
    for idx, change in zip(non_zero_indices, score_changes):
        score_change_list[idx] = change
    
    return score_change_list

def main():
    # Load the CSV file
    df = pd.read_csv('/Users/stevenwendel/Documents/GitHub/bg/minimized_df.csv')
        
    # Convert the 'dna' column from string representation to actual lists
    df['dna'] = df['dna'].apply(ast.literal_eval)

    # Process each row in the dataframe
    results = []
    for index, row in df.iterrows():
        print(index)
        score_change_list = process_dna_row(row)
        results.append((index, score_change_list))  # Save index and score change list as a tuple
    
    # Output the results
    for index, score_change_list in results:
        print(f'Index: {index}, Score Changes: {score_change_list}')

    # Optionally, save the results to a new CSV file
    results_df = pd.DataFrame(results, columns=['Index', 'ScoreChanges'])
    results_df.to_csv('/Users/stevenwendel/Documents/GitHub/bg/unique_df2.csv', index=False)

if __name__ == "__main__":
    main()
