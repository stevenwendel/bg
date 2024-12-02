import random
import numpy as np
import pandas as pd

from src.neuron import *
from src.utils import *
from src.constants import * 
from src.network import run_network
from src.validation import *
from copy import deepcopy

# def initialize_genetic_algorithm(pop_size=500, mut_rate=0.1, mut_sigma=0.3, rank_depth=None, crossover_point=None, num_gen=10, elite_passthrough=5, bounds=[0, 400], my_free_weights=None, path=''):
"""
Initializes parameters and creates the initial population for the genetic algorithm.
"""
# if rank_depth is None:
#     rank_depth = int(np.floor(pop_size / 3))
# if crossover_point is None:
#     crossover_point = int(np.floor(len(my_free_weights) / 2))

# random.seed(0)
# file_path = f'{path}pop{pop_size}_gen{num_gen}'

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
def evaluate_dna(dna_matrix, neurons, alpha_array, input_waves, criteria):
    """Evaluates a DNA weight matrix by running experimental and control network simulations.

    Runs the neural network with the given weight matrix under both experimental and control 
    conditions. Compares the spiking differences between conditions and scores the results
    according to provided criteria.

    Args:
        dna_matrix: 2D numpy array containing synaptic weights derived from a DNA sequence.
        neurons: List of Izhikevich neuron objects representing the network.
        alpha_array: Array of alpha function values for synaptic transmission.
        input_waves: List containing [cue_wave, go_wave] input signals.
        criteria: Scoring criteria for evaluating network performance.

    Returns:
        tuple containing:
            - score (float): Overall performance score based on criteria
            - neuron_data (dict): Dict containing minimal Izhikevich neurons for both conditions
                with only hist_V and spike_times attributes preserved
            - binned_differences (ndarray): Binned spike count differences between conditions
    """
    neuron_data = {'experimental': [], 'control': []}

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

        # Create minimal Izhikevich neurons with just the required attributes
        minimal_neurons = []
        for n in neurons:
            minimal_neuron = Izhikevich(name=n.name)
            minimal_neuron.hist_V = n.hist_V.copy()
            minimal_neuron.spike_times = n.spike_times.copy()
            minimal_neurons.append(minimal_neuron)
            
        neuron_data[condition] = minimal_neurons
    
    # Get differences across bins        
    binned_differences = get_binned_differences(
        experimental_neurons=neuron_data['experimental'],
        control_neurons=neuron_data['control'])

    target_binned_differences = get_neurons(binned_differences, CRITERIA_NAMES)

    # Score the results
    score = score_run(target_binned_differences, criteria)
    
    return score, neuron_data, binned_differences


def score_population(dnas, free_weights_list, pop_size, validation_neurons, experiment_criteria, control_criteria):
    """Score a population of DNA sequences by evaluating network performance.

    Evaluates each DNA sequence in the population by:
    1. Converting DNA to weight matrix
    2. Running network simulation in experimental and control conditions
    3. Calculating overall score as sum of experimental and control scores

    Args:
        dnas (list): List of DNA sequences to evaluate
        free_weights_list (list): List of synaptic weights that can be modified
        pop_size (int): Size of the population
        validation_neurons (list): List of neuron objects for validation
        experiment_criteria (dict): Scoring criteria for experimental condition
        control_criteria (dict): Scoring criteria for control condition

    Returns:
        list: List of [dna, score] pairs for each DNA sequence in population

    Example:
        >>> population_scores = score_population(dnas, weights, 100, neurons, exp_criteria, ctrl_criteria)
        >>> best_dna = max(population_scores, key=lambda x: x[1])[0]
    """
    scores = []
    for dna in dnas:
        weight_matrix = load_dna(free_weights_list, dna)
        run_network(weight_matrix, validation_neurons, sq_wave, go_wave, t_max, dt, alpha_array, control=False)
        experiment_score = score_run(validation_neurons, experiment_criteria, validation_thresholds, validation_period_times)
        run_network(weight_matrix, validation_neurons, sq_wave, go_wave, t_max, dt, alpha_array, control=True)
        control_score = score_run(validation_neurons, control_criteria, validation_thresholds, validation_period_times)
        overall_score = experiment_score + control_score
        print(f'Score: {overall_score} | DNA: {dna}')
        scores.append([dna, overall_score])
    return scores


def spawn_next_population(curr_pop: list[list[float]]) -> list[list[float]]:
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
        curr_pop (list[list[float]]): Current population as list of [dna, dna_score] pairs
            where dna is list of float weights and dna_score is float fitness score

    Returns:
        list[list[float]]: New population of DNA sequences, with length POP_SIZE

    Example:
        >>> curr_pop = [[dna1, score1], [dna2, score2], ...]
        >>> next_pop = spawn_next_population(curr_pop)
        >>> assert len(next_pop) == POP_SIZE
    """

    curr_pop.sort(key=lambda x: x['dna_score'], reverse=True)
    survivors = curr_pop[:RANK_DEPTH]
    next_dnas = [curr_pop[i]['dna'] for i in range(ELITE_SIZE)]
    
    for _ in range(POP_SIZE - ELITE_SIZE):
        parent1 = random.choice(survivors)['dna']
        parent2 = random.choice(survivors)['dna']
        child_dna=[]

        for i, synapse in enumerate(ACTIVE_SYNAPSES):
            gene = random.choice([parent1[i], parent2[i]])
            gene = random.normalvariate(gene, gene * MUT_SIGMA) if random.random() < MUT_RATE else gene
            child_dna.append(int(gene))

        next_dnas.append(child_dna)
    
    assert len(next_dnas) == POP_SIZE
    return next_dnas




# Currently unused
def run_genetic_algorithm(config, my_free_weights, validation_neurons, experiment_criteria, control_criteria, validation_thresholds, validation_period_times, t_max, dt, alpha_array, sq_wave, go_wave):
    """Runs genetic algorithm optimization of neural network weights.

    Evolves a population of weight matrices over multiple generations to optimize network behavior.
    Each generation:
    1. Scores current population by running network simulations
    2. Creates next generation through selection and mutation
    3. Saves generation results to pickle file

    Args:
        config (dict): Configuration parameters including:
            - dnas (list): Initial population of weight matrices
            - population_size (int): Size of population to maintain
            - num_generations (int): Number of generations to run
            - file_path (str): Base path for saving generation results
            - rank_depth (int): Number of top individuals used for breeding
            - mutation_rate (float): Probability of mutating each gene
            - mutation_sigma (float): Standard deviation for mutations
            - elite_passthrough (int): Number of top individuals preserved unchanged
        my_free_weights (list): List of modifiable synaptic weights
        validation_neurons (list): Neurons to validate behavior of
        experiment_criteria (DataFrame): Target behavior for experimental condition
        control_criteria (DataFrame): Target behavior for control condition
        validation_thresholds (list): Firing rate thresholds for validation
        validation_period_times (list): Time windows for validation
        t_max (int): Maximum simulation time
        dt (float): Simulation time step
        alpha_array (ndarray): Alpha function for synaptic transmission
        sq_wave (ndarray): Square wave input
        go_wave (ndarray): Go signal input

    Returns:
        list: Scores and DNA sequences for each generation

    Example:
        >>> config = {'dnas': initial_pop, 'population_size': 100, ...}
        >>> results = run_genetic_algorithm(config, weights, neurons, ...)
    """
    dnas = config['dnas']
    pop_size = config['population_size'] 
    num_gen = config['num_generations']
    file_path = config['file_path']
    gen_scores = []

    for gen in range(num_gen):
        print(f'==== Generation {gen} =====')
        scores = score_population(dnas, my_free_weights, pop_size, validation_neurons, experiment_criteria, control_criteria)
        print('Finished scoring!')
        
        pop = [[dnas[i], scores[i][1]] for i in range(pop_size)]
        gen_scores.append(pop)
        dnas = spawn_next_generation(pop, config['rank_depth'], config['mutation_rate'], config['mutation_sigma'], config['elite_passthrough'], pop_size, my_free_weights)
        
        gen_df = pd.DataFrame([[gen, individual[0], individual[1]] for individual in pop], columns=['Gen #', 'DNA', 'Score'])
        
        # Saving as .pkl to Drive
        save_path = f'{file_path}_complete.pkl' if gen == num_gen -1 else f'{file_path}_{gen+1}.pkl'
        gen_df.to_pickle(save_path)
        print(f'Saved to: {save_path}')

    return gen_scores
