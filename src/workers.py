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

def process_parent_dna(parent_dict, min_score):
    parent_dna, parent_score = parent_dict['dna'], parent_dict['dna_score']
    pid = os.getpid()
    dna_children = 0
    successful_children = []

    print(f"Process {pid} working...")    

    for i in range(len(parent_dna)):
        # Produce list of successful children data
        if parent_dna[i] != 0:
            child_dna = np.array(parent_dna)
            child_dna[i] = 0


            dna_matrix = load_dna(child_dna)
            all_neurons = create_neurons()
            _, input_waves, alpha_array = create_experiment()
            criteria_dict = define_criteria()
   
            dna_score, _ = evaluate_dna(
                dna_matrix=dna_matrix,
                neurons=all_neurons,
                alpha_array=alpha_array,
                input_waves=input_waves,
                criteria=criteria_dict,
                curr_dna=child_dna
            )
            child_score = sum(dna_score.values())
            
            if child_score >= min_score:
                print(f'Child #{dna_children} === Gene removed: {i} === New Score: {child_score} === Delta: {child_score-parent_score}')
                
                dna_children += 1
                successful_children.append({
                    'dna': tuple(child_dna), 
                    'dna_score': child_score,  
                    'is_child': True
                    })

    # Only add parent to fully_reduced if it has no valid children
    if dna_children == 0 and parent_score >= min_score:
        print(f'Parent added to fully_reduced: No valid children found')
        successful_children.append({
            'dna': tuple(parent_dna), 
            'dna_score': parent_score,
            'is_child': False,
        })

    return successful_children