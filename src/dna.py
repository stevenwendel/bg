import numpy as np
import pandas as pd
from src.utils import *
from src.neuron import *
from src.constants import * 
from src.network import *
from src.validation import *
from copy import deepcopy


def load_dna(dna: list[float]) -> np.ndarray:
    assert len(ACTIVE_SYNAPSES) == len(dna), "Number of available synapses does not match length of DNA"

    w = np.zeros((len(NEURON_NAMES), len(NEURON_NAMES)))
    for i, synapse in enumerate(ACTIVE_SYNAPSES):
        origin, termina = synapse
        
        origin_index = NEURON_NAMES.index(origin)
        termina_index = NEURON_NAMES.index(termina)
        w[origin_index, termina_index] = dna[i]
    return w


# === Running Network and Storing Results ====    
def evaluate_dna(dna_matrix, neurons, alpha_array, input_waves, criteria):
    # Note: returns binned differences of ALL neurons, not just the target ones.
    # Can use get_neurons() if you want to retrieve specific neuron differences. 
    neuron_data = {}

    for condition in ['experimental', 'control']:    

        prepare_neurons(
            neurons=neurons,
            cue_wave=input_waves[0],
            go_wave=input_waves[1],
            control=True if condition == 'control' else False
        )
        
        run_network(
            neurons=neurons,
            weight_matrix=dna_matrix, # I have this written as t_max and tMax throughout; should homogenize
            alpha_array=alpha_array,
            )
        neuron_data[condition] = deepcopy(neurons)
    
    # === Getting differences across bins ====        
    binned_differences = get_binned_differences(
        experimental_neurons=neuron_data['experimental'],
        control_neurons=neuron_data['control'])

    target_binned_differences=get_neurons(binned_differences, CRITERIA_NAMES) #CRITERIA NAMES MISSING FROM ARGUMENT

    # === Scoring ===
    score = score_run(target_binned_differences, criteria)
    
    return score, neuron_data, binned_differences 