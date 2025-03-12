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

def filter_data(original_df, completely_reduced_dna):

    # === Filter for high scoring runs ===
    # Define the dna_score threshold
    dna_threshold = 730
    # Filter the DataFrame for entries with dna_score above the threshold
    filtered_df = original_df[original_df['dna_score'] > dna_threshold].sort_values(by='dna_score', ascending=False).reset_index(drop=True)


    # === Filter for unique configurations ===
    max_synapses = 20

    ## Picking out indicies of unique configurations
    unique_configs = {}
    for index, row in filtered_df.iterrows():
        config = tuple(1 if weight != 0 else 0 for weight in row['dna'])
        # Calculate the number of non-zero entries
        non_zero_count = sum(config)
        # Only consider configurations with non-zero counts of 18 or lower
        if non_zero_count <= max_synapses and config not in unique_configs:
            # Yes, this will drop configurations that have potentially more "resilient" weights, in lieu of the maximally scoring weights, 
            # but we're not going to get everything in general, so it's okay to drop some in exchange for computational efficiency.
            
            # Store the index, dna_score, and non_zero count
            unique_configs[config] = {
                'index': index,
                'dna_score': row['dna_score'],
                'non_zero_count': non_zero_count
            }

    # Create a list of indices for the unique configurations
    unique_indices = [details['index'] for details in unique_configs.values()]

    # Create a new dataframe with only the unique configurations
    unique_df = filtered_df.loc[unique_indices].copy().reset_index(drop=True)
    print(f"Found {len(unique_df)} unique configurations")
    print(unique_df)


    # At this stage, we have a dataframe with 730+ scores, <20 synapses, and single representatives for any given unique configuration.

    # # Convert the 'dna' column from string representation to actual lists
    # df['dna'] = df['dna'].apply(ast.literal_eval)

    # This step is going to create the score change list [101,0,0,8,8,6,0,...]
    
    # Processing each row in the dataframe
    results = []
    for index, row in unique_df.iterrows():
        print(index)
        score_change_list = process_dna_row(row)
        results.append((index, score_change_list))  # Save index and score change list as a tuple
    
    # Output the results
    for index, score_change_list in results:
        print(f'Index: {index}, Score Changes: {score_change_list}')

    # Putting all those change vectors into a df. Optionally, could save the results to a CSV
    changes_df = pd.DataFrame(results, columns=['Index', 'ScoreChanges'])


    # === Create new dataframe where indices of subthreshold score changes are set to 0 ===

    # Define the threshold
    threshold = 5

    # List to store minimized DNA
    minimized_dna_list = []
    assert len(unique_df) == len(changes_df)
    # Iterate over each row in the dataframes
    for i in range(len(unique_df)):
        # Get the DNA and changes arrays
        dna = unique_df.iloc[i]['dna']
        changes = changes_df.iloc[i]['ScoreChanges']
        
        # Check if changes is a string and needs to be evaluated
        if isinstance(changes, str):
            changes = ast.literal_eval(changes)  # Use ast.literal_eval for safety
        
        # Ensure all elements are integers 
        changes = list(map(int, changes))
        


        ## Save fully minimized DNAs (i.e. nothing is below threshold for them)
        # Check if all changes are either above threshold or 0
        if all(change == 0 or change > threshold for change in changes):
            # Save this DNA to fully minimal list since no more reductions needed
            completely_reduced_dna = pd.concat([completely_reduced_dna, unique_df.iloc[i]])
            continue # Skip to next DNA since this one is fully reduced

        # Create a new DNA list with elements set to 0 if the corresponding change is below the threshold
        for j in range(len(dna)):
            if (changes[j] <= threshold) and (dna[j] != 0):
                minimized_dna = dna.copy()
                minimized_dna[j] = 0
                # Should produce a list of once-changed, unevaluated DNAs from the parents. 
                # What happens to the parents who have no children (are fully reduced)?
            
                # Append the minimized DNA to the list
                minimized_dna_list.append({'dna': minimized_dna,
                                        'dna_score': unique_df.iloc[i]['dna_score'] - changes[j]})

        # At this step, minimized_dna_list is full of DNA to be rerun, and see if any of them still work.
        # the ones that do still work will be filtered; their deltas will be calculated; and repeat
  

    # Return the list as a DataFrame
    return pd.DataFrame(minimized_dna_list)

def main():
    start_time = time.time()
    # === Load the combined data ===
    # Load the combined data from the .pkl file
    with open('/Users/stevenwendel/Documents/GitHub/bg/combined_data.pkl', 'rb') as file:
        combined_data = pickle.load(file)

    # Create a DataFrame from the combined data
    untested_dna = pd.DataFrame(combined_data)
    iteration = 0  # Initialize iteration counter
    
    # Initialize lists to store fully reduced DNAs
    completely_reduced_dna = pd.DataFrame()
    

    while len(untested_dna) > 0:
        
        print(f'Iteration: {iteration} ==== Current untestedsize: {len(untested_dna)}')
        untested_dna = filter_data(untested_dna, completely_reduced_dna)

        # Save the minimized DataFrame to a CSV file
        untested_dna.to_csv(f'untested_data_pass_{iteration}.csv', index=False)
        completely_reduced_dna.to_csv(f'completely_reduced_data_pass_{iteration}.csv', index=False)
          
        iteration += 1  # Increment iteration counter
    
    print(f'Time taken: {time.time() - start_time:.2f} seconds')

if __name__ == "__main__":
    main()

#Should I not be dropping the index? it could serve as a barcode to see what is kept and what is dropped. Might fuck up my ilocs though.