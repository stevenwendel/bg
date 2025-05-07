import random
import numpy as np
import pandas as pd
import copy
from typing import List, Dict, Any
import time

from src.neuron import Izhikevich, prepare_neurons, create_neurons
from src.utils import *
from src.constants import * 
from src.network import run_network, create_experiment
from src.validation import define_criteria, calculate_score
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
def evaluate_dna(dna_matrix, neurons, alpha_array, input_waves, criteria):

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
    """Evaluate a single DNA sequence in a separate process.
    
    Args:
        args: Tuple containing:
            - curr_dna: The DNA sequence to evaluate
            - all_neurons: List of neuron templates
            - alpha_array: Alpha array for synaptic dynamics
            - input_waves: Input wave patterns
            - criteria_dict: Criteria for evaluation
            - generation: Current generation number
            - max_score: Maximum possible score
            
    Returns:
        Tuple of (curr_dna, total_score)
    """
    curr_dna, all_neurons, alpha_array, input_waves, criteria_dict, generation, max_score = args
    
    # Create a deep copy of neurons for this process to avoid shared state
    process_neurons = [copy.deepcopy(neu) for neu in all_neurons]
    
    # Loading dna into matrix
    dna_matrix = load_dna(curr_dna) 

    # Running network to score dna
    dna_scores, neuron_data = evaluate_dna(
        dna_matrix=dna_matrix,
        neurons=process_neurons,
        alpha_array=alpha_array,
        input_waves=input_waves,
        criteria=criteria_dict
        )
    
    total_score = sum(dna_scores.values())

    # print(f'Gen {generation} === Overall: {total_score}') 

    # Clean up process-specific data
    del process_neurons
    del dna_matrix
    del neuron_data

    return curr_dna, total_score

def spawn_next_population(curr_pop: list[dict], ga_config: dict, generation: int) -> tuple[list[list[float]], dict]:
    """Generates the next population of DNA sequences through selection and mutation.

    Creates a new population by:
    1. Selecting top performing individuals as survivors based on RANK_DEPTH
    2. Preserving ELITE_SIZE best individuals unchanged
    3. Creating remaining individuals through:
        - Tournament selection of parents
        - Adaptive mutation based on population diversity
        - Smart crossover with diversity preservation

    Args:
        curr_pop (list[dict]): Current population as list of dictionaries
        ga_config (dict): Configuration parameters for the GA
        generation (int): Current generation number

    Returns:
        tuple: (next_dnas, stats_dict) where:
            - next_dnas: New population of DNA sequences
            - stats_dict: Dictionary containing mutation statistics
    """
    curr_pop.sort(key=lambda x: x['dna_score'], reverse=True)
    survivors = curr_pop[:ga_config['RANK_DEPTH']]
    next_dnas = [curr_pop[i]['dna'] for i in range(ga_config['ELITE_SIZE'])]
    
    # Calculate population diversity
    def calculate_diversity(population):
        if not population:
            return 0
        # Calculate average pairwise distance between DNA sequences
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = sum(abs(a - b) for a, b in zip(population[i], population[j]))
                distances.append(dist)
        return sum(distances) / len(distances) if distances else 0

    # Adaptive mutation parameters
    base_mutation_rate = ga_config['MUT_RATE']
    base_mutation_sigma = ga_config['MUT_SIGMA']
    diversity = calculate_diversity([p['dna'] for p in survivors])
    max_diversity = ga_config['DNA_BOUNDS'][1] * len(ACTIVE_SYNAPSES) * 0.5  # Theoretical max diversity
    
    # Calculate actual mutation parameters
    mutation_rate = base_mutation_rate  # * (diversity/max_diversity)
    mutation_sigma = base_mutation_sigma * (1.5 - diversity/max_diversity)
    
    # Store statistics
    stats = {
        'diversity': diversity,
        'max_diversity': max_diversity,
        'mutation_rate': mutation_rate,
        'mutation_sigma': mutation_sigma,
        'diversity_ratio': diversity/max_diversity
    }
    
    # Tournament selection
    def tournament_select(population, tournament_size=3):
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x['dna_score'])['dna']

    # Smart crossover
    def smart_crossover(parent1, parent2):
        child = []
        boundary = ga_config['DNA_BOUNDS'][1]  # Move boundary definition to top
        
        for i, synapse in enumerate(ACTIVE_SYNAPSES):
            # Prefer genes from better parent with 60% probability
            if random.random() < 0.6:
                gene = parent1[i]
            else:
                gene = parent2[i]
            
            # Adaptive mutation - higher rate when diversity is high (early generations)
            if random.random() < mutation_rate:
                # Gentle encouragement towards zero: scale sigma by |gene|/boundary
                # This means larger values have more room to mutate, but smaller values are more stable
                zero_attraction = abs(gene) / boundary
                sigma = gene * mutation_sigma * (1 + zero_attraction) * (1 - diversity/max_diversity)
                
                # Add a small bias towards zero
                # The closer we are to zero, the more likely we stay there
                if random.random() < zero_attraction * 0.2:  # 20% max chance of moving towards zero
                    gene = gene * (1 - random.random() * 0.4)  # Reduce by up to 40%
                
                # Apply the mutation
                gene = random.normalvariate(gene, sigma=sigma)
            
            # Bounding DNA
            gene = max(min(gene, boundary), -boundary)
            
            # Ensure correct sign for inhibitory neurons
            if synapse[0] in INHIBITORY_NEURONS:
                gene = -abs(gene)
            else:
                gene = abs(gene)
            
            child.append(int(gene))
        return child

    # Generate new population with early stopping
    max_attempts = ga_config['POP_SIZE'] * 2  # Limit attempts to prevent infinite loop
    attempts = 0
    
    while len(next_dnas) < ga_config['POP_SIZE'] and attempts < max_attempts:
        parent1 = tournament_select(survivors)
        parent2 = tournament_select(survivors)
        child_dna = smart_crossover(parent1, parent2)
        
        if child_dna not in next_dnas:
            next_dnas.append(child_dna)
        attempts += 1
    
    # If we couldn't generate enough unique children, fill with mutations of best individuals
    if len(next_dnas) < ga_config['POP_SIZE']:
        print(f"Warning: Could only generate {len(next_dnas)} unique children. Filling with mutations.")
        while len(next_dnas) < ga_config['POP_SIZE']:
            parent = random.choice(survivors)['dna']
            child_dna = smart_crossover(parent, parent)  # Self-crossover with mutation
            if child_dna not in next_dnas:
                next_dnas.append(child_dna)

    assert len(next_dnas) == ga_config['POP_SIZE']
    return next_dnas, stats

def run_genetic_algorithm(ga_config: dict, neurons: List[Izhikevich], ga_set: str) -> List[Dict[str, Any]]:
    """Runs the genetic algorithm to evolve neural network weights.

    Args:
        ga_config (dict): Configuration parameters for the GA
        neurons (List[Izhikevich]): List of neurons to evaluate
        ga_set (str): Name of the GA run

    Returns:
        List[Dict[str, Any]]: Final population with their scores
    """
    # Early stopping parameters
    patience = 20  # Number of generations to wait for improvement
    min_improvement = 0.001  # Minimum improvement threshold
    best_score_history = []
    no_improvement_count = 0
    
    # Convergence tracking
    diversity_history = []
    convergence_threshold = 0.1  # Population is considered converged when diversity drops below this
    
    # Initialize population
    curr_population = [{'dna': create_dna(ga_config['DNA_BOUNDS']), 'dna_score': None} 
                      for _ in range(ga_config['POP_SIZE'])]
    
    save_dict = {
        'metadata': {
            'ga_set': ga_set,
            'config': ga_config,
            'start_time': time.time(),
            'generation': 0
        },
        'generations': []
    }
    
    for generation in range(ga_config['NUM_GENERATIONS']):
        print(f"\nGeneration {generation}")
        
        # Evaluate current population
        population_results = []
        for dna_dict in curr_population:
            dna_matrix = load_dna(dna_dict['dna'])
            scores, _ = evaluate_dna(
                dna_matrix=dna_matrix,
                neurons=neurons,
                alpha_array=alpha_array,
                input_waves=input_waves,
                criteria=criteria_dict
            )
            total_score = sum(scores.values())
            population_results.append({
                'dna': dna_dict['dna'],
                'dna_score': total_score
            })
        
        # Sort by score
        population_results.sort(key=lambda x: x['dna_score'], reverse=True)
        curr_population = population_results
        
        # Track best score
        best_score = curr_population[0]['dna_score']
        best_score_history.append(best_score)
        
        # Calculate population diversity
        diversity = calculate_population_diversity(curr_population)
        diversity_history.append(diversity)
        
        # Check for convergence
        if diversity < convergence_threshold:
            print(f"Population converged at generation {generation} (diversity: {diversity:.3f})")
            break
            
        # Early stopping check
        if len(best_score_history) > 1:
            improvement = best_score - best_score_history[-2]
            if improvement < min_improvement:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print(f"Early stopping at generation {generation} - No improvement for {patience} generations")
                    break
            else:
                no_improvement_count = 0
        
        # Save generation results
        save_dict['generations'].append({
            'generation': generation,
            'best_score': best_score,
            'avg_score': sum(p['dna_score'] for p in curr_population) / len(curr_population),
            'diversity': diversity,
            'population': curr_population
        })
        
        # Print progress
        print(f"Best score: {best_score:.3f}")
        print(f"Average score: {sum(p['dna_score'] for p in curr_population) / len(curr_population):.3f}")
        print(f"Population diversity: {diversity:.3f}")
        
        # Generate next population
        next_dnas, stats = spawn_next_population(curr_population, ga_config, generation)
        curr_population = [{'dna': dna, 'dna_score': None} for dna in next_dnas]
    
    return curr_population

def calculate_population_diversity(population: list[dict]) -> float:
    """Calculate the diversity of the population based on DNA differences.
    
    Args:
        population (list[dict]): Current population
        
    Returns:
        float: Population diversity score between 0 and 1
    """
    if not population:
        return 0.0
        
    # Calculate average pairwise distance between DNA sequences
    distances = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            dist = sum(abs(a - b) for a, b in zip(population[i]['dna'], population[j]['dna']))
            distances.append(dist)
    
    # Normalize diversity score
    max_possible_distance = len(population[0]['dna']) * 2  # Assuming DNA values are between -1 and 1
    return sum(distances) / (len(distances) * max_possible_distance) if distances else 0.0
