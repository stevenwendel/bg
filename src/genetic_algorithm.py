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
        if synapse[0] in INHIBITORY_NEURONS:
            dna_val *= -1
        dna.append(dna_val)
    return dna



def load_dna(dna: list[float]) -> np.ndarray:
    assert len(ACTIVE_SYNAPSES) == len(dna), "Number of available synapses does not match length of DNA"
    w = np.zeros((len(NEURON_NAMES), len(NEURON_NAMES)))
    connection_count = {}  # Dictionary to keep track of added connections

    for i, synapse in enumerate(ACTIVE_SYNAPSES):
        origin, target = synapse
        
        try:
            origin_index = NEURON_NAMES.index(origin)
            target_index = NEURON_NAMES.index(target)
        except ValueError as e:
            raise ValueError(f"Invalid synapse '{synapse}': {e}")

        # If the connection has already been set, warn and sum the contribution
        if (origin_index, target_index) in connection_count:
            print(f"Warning: Duplicate connection {origin} -> {target} encountered; summing weights.")
        w[origin_index, target_index] += dna[i]
        connection_count[(origin_index, target_index)] = connection_count.get((origin_index, target_index), 0) + 1

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
    
        critical_neurons_spikes = np.array([neuron_data[condition][name]['spike_times'] for name in CRITERIA_NAMES])
        critical_neuron_spike_bins = np.reshape(critical_neurons_spikes, (len(CRITERIA_NAMES), TMAX//BIN_SIZE, BIN_SIZE) #I think this is a problem... how does Sum work?
                ).sum(axis=2)
        critical_neuron_criteria = criteria[condition]

        scores[condition] = calculate_score(critical_neuron_spike_bins, critical_neuron_criteria, condition)  #should add curr_dna to arguments...
    
    return scores, neuron_data

def drone_evaluate_dna(args):
    curr_dna, all_neurons, alpha_array, input_waves, criteria_dict, generation, max_score = args
    
    # Loading dna into matrix
    dna_matrix = load_dna(curr_dna) 

    # Running network to score dna
    dna_scores, neuron_data = evaluate_dna(
        dna_matrix=dna_matrix,
        neurons=all_neurons,
        alpha_array=alpha_array,
        input_waves=input_waves,
        criteria=criteria_dict,
        curr_dna=curr_dna
        )
    
    total_score = sum(dna_scores.values())

    # print(f'Gen {generation} === DNA: {curr_dna}') 
    # print(f'    === Control: {dna_scores["control"]}/{max_score}')
    # print(f'    === Experimental: {dna_scores["experimental"]}/{max_score}')
    # print(f'    === Overall: {total_score}({total_score/(2*max_score):.2%})\n')
    print(f'{generation=} === {total_score=}')

    return curr_dna, total_score

def spawn_next_population(curr_pop: list[dict], ga_config: dict, generation: int, gen_max_scores: list) -> list[list[float]]:
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
    
    while len(next_dnas) < ga_config['POP_SIZE']:
        boundary = ga_config['DNA_BOUNDS'][1]
        parent1 = random.choice(survivors)['dna']
        parent2 = random.choice(survivors)['dna']
        child_dna=[]

        # Construct child dna
        for i, synapse in enumerate(ACTIVE_SYNAPSES):

            gene = random.choice([parent1[i], parent2[i]])

            """Sigma is a square root function, and it might be growing too large (and making too many changes) around gen 500, 
            so that only a few get propagated.
            I can either remove the function...
            sigma = ga_config['MUT_SIGMA'] * (gene * (generation / ga_config['NUM_GENERATIONS'])**(1/2))            
            OR set a uniqueness condition on the dna, such that no dna may be passed along twice, i.e. each dna must be unique
            """ 
            period = 100
            sigma = ga_config['MUT_SIGMA'] * gene * np.sin(generation/period)**2       
            gene = random.normalvariate(gene, sigma) if random.random() < ga_config['MUT_RATE'] else gene

            # Introduce jitter to allow zeroed-out genes to potentially become non-zero
            unjittered_generations = np.min([ga_config['NUM_GENERATIONS'], 100])
            # assert unjittered_generations < ga_config['NUM_GENERATIONS']
            
            decay_rate = 1/2
            if generation < ga_config['NUM_GENERATIONS']-unjittered_generations:
                jitter = random.uniform(-1, 1) * (ga_config['DNA_BOUNDS'][1] * 0.05) * .992 ** generation  # Adjust the scale of jitter as needed            
                gene += jitter

            # Bounding DNA
            if abs(gene) > boundary:
                gene = boundary

            # Inhibitory neurons have negative weights
            if synapse[0] in INHIBITORY_NEURONS:
                gene = -abs(gene)
            elif synapse[0] not in INHIBITORY_NEURONS:
                gene = abs(gene)
            else:
                raise ValueError(f'Invalid synapse: {synapse}')
            
            child_dna.append(int(gene))

            # if len(gen_max_scores) > 10:
            #     if gen_max_scores[-10] == gen_max_scores[-1] and random.random() < .5:
            #         child_dna = [int(gene + np.random.normal(0, 50)) for gene in child_dna]

        # Only add child_dna if not already passing on
        if child_dna not in next_dnas:
            next_dnas.append(child_dna)

    assert len(next_dnas) == ga_config['POP_SIZE']
    return next_dnas
