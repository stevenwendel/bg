import random
import numpy as np
import pandas as pd

from src.neuron import *
from src.utils import *
from src.constants import * 
from src.network import run_network
from src.validation import *
from copy import deepcopy

from scipy.linalg import norm



def create_dna(bounds: list[float]) -> list[float]:
    """Creates a single strand of DNA representing synaptic weights for a neural network.

    Generates random weight values within the specified bounds for each synapse in ACTIVE_SYNAPSES.
    For inhibitory neurons (SNR, MSN, ALMinter), the weights are made negative.

    Args:
        bounds: A list of [min, max] values specifying the allowed range for weight values.

    Returns:
        A list of float values representing the synaptic weights, with length equal to 
        the number of active synapses. Weights for inhibitory neurons are negative.

    Example:
        >>> dna = create_dna([0, 400])
        >>> len(dna) == len(ACTIVE_SYNAPSES)
        True
    """
    dna = []
    for synapse in ACTIVE_SYNAPSES:
        dna_val = random.randint(bounds[0], bounds[1])
        if synapse[0] in ["SNR1","SNR2", "SNR3", "MSN1", "MSN2", "MSN3", "ALMinter"]:
            dna_val *= -1
        dna.append(dna_val)
    return dna



def load_dna(dna: list[float]) -> np.ndarray:
    """Converts a DNA sequence of synaptic weights into a weight matrix for a neural network.

    Takes a list of synaptic weights (DNA) and creates a weight matrix where each entry w[i,j] 
    represents the connection strength from neuron i to neuron j. Only synapses specified in 
    ACTIVE_SYNAPSES are populated, all others remain 0.

    Args:
        dna: A list of float values representing synaptic weights. Must have same length as 
            ACTIVE_SYNAPSES. Each value corresponds to the weight of one synapse.

    Returns:
        A 2D numpy array of shape (n_neurons, n_neurons) containing the weight matrix, where
        n_neurons is the number of neurons in NEURON_NAMES. Matrix entry w[i,j] represents 
        the connection strength from neuron i to neuron j.

    Raises:
        AssertionError: If length of dna does not match number of active synapses.

    Example:
        >>> dna = [1.0, -2.0, 3.0] # Weights for 3 synapses
        >>> w = load_dna(dna)
        >>> w.shape == (len(NEURON_NAMES), len(NEURON_NAMES))
        True
    """

    assert len(ACTIVE_SYNAPSES) == len(dna), "Number of available synapses does not match length of DNA"

    w = np.zeros((len(NEURON_NAMES), len(NEURON_NAMES)))
    for i, synapse in enumerate(ACTIVE_SYNAPSES):
        origin, termina = synapse
        
        origin_index = NEURON_NAMES.index(origin)
        termina_index = NEURON_NAMES.index(termina)
        w[origin_index, termina_index] = dna[i]
    return w



# === Running Network and Storing Results ====    
def evaluate_dna(dna_matrix, neurons, alpha_array, input_waves, criteria, curr_dna):

    neuron_data = {}
    scores = {}

    for condition in ['experimental', 'control']:    
        prepare_neurons(
            neurons=neurons,
            cue_wave=input_waves[0],
            go_wave=input_waves[1],
            control=True if condition == 'control' else False
        )
        
        run_network(
            neurons=neurons,
            weight_matrix=dna_matrix,
            alpha_array=alpha_array,
        )

        # Neurons have been run and loaded with information. Should I reset them after this?

        condition_data = {}
        for n in neurons:
            condition_data[n.name] = {
                'hist_V': n.hist_V.copy(),
                'spike_times': n.spike_times.copy()
            }
        neuron_data[condition] = condition_data
    
        target_neurons_spikes = np.array([neuron_data[condition][name]['spike_times'] for name in CRITERIA_NAMES])
        target_neuron_spike_bins = np.reshape(target_neurons_spikes, (len(CRITERIA_NAMES), TMAX//BIN_SIZE, BIN_SIZE)
                ).sum(axis=2)
        target_neuron_criteria = criteria[condition]
        # Calculate the score

        # # Normalize score by L1 norm of current DNA
        # l1_norm = norm(curr_dna, 1) 
        # l1_norm_transform = l1_norm
        # print(f'L1 norm: {l1_norm}')

        # def normal_dist(x, mean, sd):
        #     prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
        #     return prob_density

        #Indicator function for zero weights; scaled
        # bins_totes = target_neuron_spike_bins.shape[1] *2 #mulitiplied by 2 to account for control and experimental; 
        # mui= bins_totes * 1/target_neuron_spike_bins.shape[0] # percent of bins I'm willing to sacrifice for a zero weight
        # zero_count = sum(mui for gene in curr_dna if abs(gene) < 5)

        scores[condition] = calculate_score(target_neuron_spike_bins, target_neuron_criteria)  #should add curr_dna to arguments...
    
    return scores, neuron_data



def spawn_next_population(curr_pop: list[dict], ga_config: dict) -> list[list[float]]:
    """Generates the next population of DNA sequences through selection and mutation.

    Creates a new population by:
    1. Selecting top performing individuals as survivors based on RANK_DEPTH
    2. Preserving ELITE_SIZE best individuals unchanged
    3. Creating remaining individuals through:
        - Random selection of two parents from survivors
        - Random inheritance of genes from parents
        - Random mutation of genes with probability MUT_RATE
        - Mutation drawn from normal distribution with std=gene*MUT_SIGMA

    Args:
        curr_pop (list[dict]): Current population as list of dictionaries, where each dict
            contains:
            - 'dna': list[float] - list of synaptic weights
            - 'dna_score': float - fitness score for this DNA sequence

    Returns:
        list[list[float]]: New population of DNA sequences, with length POP_SIZE

    Example:
        >>> curr_pop = [{'dna': [1.0, 2.0], 'dna_score': 0.8}, ...]
        >>> next_pop = spawn_next_population(curr_pop)
        >>> assert len(next_pop) == POP_SIZE
    """

    curr_pop.sort(key=lambda x: x['dna_score'], reverse=True)
    survivors = curr_pop[:ga_config['RANK_DEPTH']]
    next_dnas = [curr_pop[i]['dna'] for i in range(ga_config['ELITE_SIZE'])]
    
    for _ in range(ga_config['POP_SIZE'] - ga_config['ELITE_SIZE']):
        parent1 = random.choice(survivors)['dna']
        parent2 = random.choice(survivors)['dna']
        child_dna=[]

        for i, synapse in enumerate(ACTIVE_SYNAPSES):
            gene = random.choice([parent1[i], parent2[i]])
            gene = random.normalvariate(gene, gene * ga_config['MUT_SIGMA']) if random.random() < ga_config['MUT_RATE'] else gene
            child_dna.append(int(gene))

        next_dnas.append(child_dna)
    
    assert len(next_dnas) == ga_config['POP_SIZE']
    return next_dnas