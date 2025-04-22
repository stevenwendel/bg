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

def process_parent_dna(parent_dict, min_score, unique_dna_sequences) -> list[dict[list[int],int,bool]]:
    pid = os.getpid()
    print(f"Process {pid} working...")    

    parent_dna, old_parent_score = parent_dict['dna'], parent_dict['dna_score']
    dna_children = 0
    successful_children = []

    new_parent_score = run_experiment(parent_dict['dna'],[0,0,0,0])

    if new_parent_score < min_score:
        print(f'faulty parent ~~ original score: {old_parent_score} ~~ real score: {new_parent_score}')
        return #[]?

    for i in range(len(parent_dna)):
        # Produce list of successful children data
        if parent_dna[i] != 0:
            child_dna = np.array(parent_dna)
            child_dna[i] = 0

            config_mask = tuple(1 if abs(i) > 0 else 0 for i in child_dna)
            if config_mask in unique_dna_sequences:
                continue

            child_score = run_experiment(child_dna,[0,0,0,0])
            
            if child_score >= min_score:
                print(f' Found child #{dna_children}! == Gene: {i} == Child Score: {child_score} == Delta: {child_score-new_parent_score}')
                
                dna_children += 1
                successful_children.append({
                    'dna': tuple(child_dna), 
                    'dna_score': child_score,  
                    'is_child': True
                    })

    # Only add parent to fully_reduced if it has no valid children
    if dna_children == 0:
        print(f'Parent added to fully_reduced: No valid children found')
        successful_children.append({
            'dna': parent_dict['dna'],  
            'dna_score': new_parent_score,
            'is_child': False,
        })

    return successful_children