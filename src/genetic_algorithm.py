import random
import numpy as np
import pandas as pd
import copy
from typing import List, Dict, Any
import time
import sys, os
import gc
import pickle
import matplotlib.pyplot as plt
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
    survivors = curr_pop[:(ga_config['POP_SIZE']//2)]
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
    mutation_rate = base_mutation_rate   * (1 + base_mutation_rate - diversity/max_diversity)
    mutation_sigma = base_mutation_sigma * (1 + base_mutation_sigma - diversity/max_diversity)
    
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
                if random.random() < zero_attraction * 0.1:  # 20% max chance of moving towards zero
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
    same_score_count = 0  # Track how long we've had the same best score
    REL_TOL = 1e-3
    last_best_score = float('-inf')
    
    # Convergence tracking
    diversity_history = []
    convergence_threshold = .05  # Population is considered converged when diversity drops below this
    
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
        
        # Check if we're stuck at the same score
        if abs(best_score - last_best_score) < REL_TOL * max(1, abs(last_best_score)):
            same_score_count += 1
        else:
            same_score_count = 0
        last_best_score = best_score
        
        # Calculate population diversity
        diversity = calculate_population_diversity(curr_population,ga_config['DNA_BOUNDS'])
        diversity_history.append(diversity)
        
        # Check for convergence and inject diversity if needed
        if diversity < convergence_threshold or same_score_count >= 10:  # Added same_score_count check
            print(f"Population converged or stuck at same score for {same_score_count} generations")
            print("Injecting diversity into population...")
            
            # Keep top 20% of population
            num_elites = max(1, int(ga_config['POP_SIZE'] * 0.2))
            elites = curr_population[:num_elites]
            
            # Create new random individuals for 30% of population
            num_new = int(ga_config['POP_SIZE'] * 0.3)
            new_individuals = []
            for _ in range(num_new):
                new_dna = create_unique_dna(ga_config['DNA_BOUNDS'], elites + new_individuals)
                new_individuals.append({'dna': new_dna, 'dna_score': None})
            
            # Create mutated versions of elites for remaining 50%
            num_mutated = ga_config['POP_SIZE'] - num_elites - num_new
            mutated = []
            for elite in elites:
                # Create multiple mutated versions of each elite
                for _ in range(num_mutated // num_elites):
                    # Apply stronger mutation to elite
                    mutated_dna = []
                    for gene in elite['dna']:
                        if random.random() < 0.5:  # 50% chance of mutation
                            # Stronger mutation when diversity is low
                            mutation = random.normalvariate(0, ga_config['MUT_SIGMA'] * 3)  # Increased from 2 to 3
                            gene = gene + mutation
                            # Ensure gene stays within bounds
                            gene = max(min(gene, ga_config['DNA_BOUNDS'][1]), -ga_config['DNA_BOUNDS'][1])
                        mutated_dna.append(int(gene))
                    
                    # Only add if unique
                    if not is_duplicate_dna(mutated_dna, elites + new_individuals + mutated):
                        mutated.append({'dna': mutated_dna, 'dna_score': None})
            
            # Combine all individuals
            curr_population = elites + new_individuals + mutated
            print(f"Injected {len(new_individuals)} new random individuals and {len(mutated)} mutated elites")
            same_score_count = 0  # Reset the counter after injecting diversity
            continue
            
        # Early stopping check - inject diversity if no improvement
        if len(best_score_history) > 5:
            # Calculate improvement over last 5 generations
            recent_scores = best_score_history[-5:]
            improvement = best_score - min(recent_scores[:-1])  # Compare with worst of previous 4 scores
            print(f"Best score: {best_score:.3f}")
            print(f"Worst recent score: {min(recent_scores[:-1]):.3f}")
            print(f"Improvement over recent window: {improvement:.6f}")
            print(f"No improvement count: {no_improvement_count}")
            
            if improvement < min_improvement:
                no_improvement_count += 1
                print(f"Increased no_improvement_count to {no_improvement_count}")
                
                if no_improvement_count >= patience:
                    print(f"No improvement for {patience} generations - Injecting diversity...")
                    
                    # Keep top 10% of population
                    num_elites = max(1, int(ga_config['POP_SIZE'] * 0.1))
                    elites = curr_population[:num_elites]
                    
                    # Create new random individuals for 40% of population
                    num_new = int(ga_config['POP_SIZE'] * 0.4)
                    new_individuals = []
                    for _ in range(num_new):
                        new_dna = create_unique_dna(ga_config['DNA_BOUNDS'], elites + new_individuals)
                        new_individuals.append({'dna': new_dna, 'dna_score': None})
                    
                    # Create mutated versions of elites for remaining 50%
                    num_mutated = ga_config['POP_SIZE'] - num_elites - num_new
                    mutated = []
                    for elite in elites:
                        # Create multiple mutated versions of each elite
                        for _ in range(num_mutated // num_elites):
                            # Apply stronger mutation to elite
                            mutated_dna = []
                            for gene in elite['dna']:
                                if random.random() < 0.7:  # 70% chance of mutation
                                    # Stronger mutation when stuck
                                    mutation = random.normalvariate(0, ga_config['MUT_SIGMA'] * 4)  # Increased from 3 to 4
                                    gene = gene + mutation
                                    # Ensure gene stays within bounds
                                    gene = max(min(gene, ga_config['DNA_BOUNDS'][1]), -ga_config['DNA_BOUNDS'][1])
                                mutated_dna.append(int(gene))
                            
                            # Only add if unique
                            if not is_duplicate_dna(mutated_dna, elites + new_individuals + mutated):
                                mutated.append({'dna': mutated_dna, 'dna_score': None})
                    
                    # Combine all individuals
                    curr_population = elites + new_individuals + mutated
                    print(f"Injected {len(new_individuals)} new random individuals and {len(mutated)} mutated elites")
                    no_improvement_count = 0
                    same_score_count = 0  # Reset the counter after injecting diversity
                    continue
            else:
                print("Resetting no_improvement_count to 0 due to improvement")
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

def calculate_population_diversity(population: list[dict],
                                   bounds: list[float]) -> float:
    """
    Diversity = mean pair‑wise Manhattan distance,
    normalised by the maximum possible distance.
    Returns value in [0, 1].
    """
    if len(population) < 2:
        return 0.0

    # pair‑wise L1 distances
    distances = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            d = sum(abs(a - b) for a, b in
                    zip(population[i]['dna'], population[j]['dna']))
            distances.append(d)

    max_dist = len(population[0]['dna']) * (bounds[1] - bounds[0])  # 800 here
    return np.mean(distances) / max_dist

def is_duplicate_dna(dna: list[float], population: list[dict], tolerance: float = 0.01) -> bool:
    """Check if a DNA sequence is too similar to any existing DNA in the population.
    
    Args:
        dna: The DNA sequence to check
        population: Current population
        tolerance: Maximum allowed difference to be considered a duplicate
        
    Returns:
        bool: True if DNA is too similar to any existing DNA
    """
    for individual in population:
        # Calculate normalized difference between DNAs
        diff = sum(abs(a - b) for a, b in zip(dna, individual['dna'])) / len(dna)
        if diff < tolerance:
            return True
    return False

def create_unique_dna(bounds: list[float], population: list[dict], max_attempts: int = 100) -> list[float]:
    """Create a new unique DNA sequence that's not too similar to existing ones.
    
    Args:
        bounds: DNA value bounds
        population: Current population
        max_attempts: Maximum number of attempts to create unique DNA
        
    Returns:
        list[float]: New unique DNA sequence
    """
    for _ in range(max_attempts):
        new_dna = create_dna(bounds)
        if not is_duplicate_dna(new_dna, population):
            return new_dna
    # If we couldn't create a unique DNA, return the last attempt
    return new_dna



# Create a means to LOAD a run
def load_ga_run(file_path):
    """Load a saved genetic algorithm run and return the data.
    
    Args:
        file_path (str): Path to the pickle file containing the GA run data
        
    Returns:
        dict: The loaded data containing all generations and metadata
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def analyze_ga_run(data):
    """Analyze a loaded genetic algorithm run.
    
    Args:
        data (dict): The loaded GA run data
        
    Returns:
        dict: Analysis results including statistics for each generation
    """
    if not data:
        return None
        
    analysis = {
        'metadata': data['metadata'],
        'generations': {}
    }
    
    # Analyze each generation
    for gen_key in [k for k in data.keys() if k.startswith('gen_')]:
        gen_num = int(gen_key.split('_')[1])
        population = data[gen_key]['population']
        
        # Extract scores and DNA
        scores = [p['dna_score'] for p in population]
        dnas = [p['dna'] for p in population]
        
        # Calculate statistics
        analysis['generations'][gen_num] = {
            'timestamp': data[gen_key]['timestamp'],
            'population_size': len(population),
            'score_stats': {
                'min': min(scores),
                'max': max(scores),
                'mean': sum(scores) / len(scores),
                'std': np.std(scores) if len(scores) > 1 else 0
            },
            'top_5_scores': sorted(scores, reverse=True)[:5],
            'top_5_dnas': [dnas[i] for i in np.argsort(scores)[-5:][::-1]]
        }
    
    return analysis

def plot_ga_progress(analysis):
    """Plot the progress of the genetic algorithm run.
    
    Args:
        analysis (dict): The analysis results from analyze_ga_run
    """
    if not analysis:
        return
        
    generations = sorted(analysis['generations'].keys())
    max_scores = [analysis['generations'][g]['score_stats']['max'] for g in generations]
    mean_scores = [analysis['generations'][g]['score_stats']['mean'] for g in generations]
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, max_scores, 'b-', label='Max Score')
    plt.plot(generations, mean_scores, 'r--', label='Mean Score')
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.title('GA Progress')
    plt.legend()
    plt.grid(True)
    plt.show()
