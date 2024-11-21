import random
import numpy as np
import pandas as pd

from src.neuron import *
from src.utils import *
from src.constants import * 
from src.dna import *
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
    """ Creates a single strand of DNA within the given bounds for all synapses"""
    dna = []
    for synapse in ACTIVE_SYNAPSES:
        dna_val = random.randint(bounds[0], bounds[1])
        if synapse[0] in ["SNR1","SNR2", "SNR3", "MSN1", "MSN2", "MSN3", "ALMinter"]:
            dna_val *= -1
        dna.append(dna_val)
    return dna

def spawn_next_population(curr_pop: list[list[float]]) -> list[list[float]]:
    """
    current pop is {dna, dna_score}
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
            gene = random.normalvariate(gene, MUT_SIGMA) if random.random() < MUT_RATE else gene
            child_dna.append(int(gene))

        next_dnas.append(child_dna)
    
    assert len(next_dnas) == POP_SIZE
    return next_dnas

def score_population(dnas, free_weights_list, pop_size, validation_neurons, experiment_criteria, control_criteria):
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

# # Create initial population
# dnas = [create_dna(bounds) for _ in range(POP_SIZE)]
# return {
#     "dnas": dnas,
#     "file_path": file_path,
#     "mutation_rate": mut_rate,
#     "mutation_sigma": mut_sigma,
#     "population_size": pop_size,
#     "crossover_point": crossover_point,
#     "num_generations": num_gen,
#     "elite_passthrough": elite_passthrough
# }

def run_genetic_algorithm(config, my_free_weights, validation_neurons, experiment_criteria, control_criteria, validation_thresholds, validation_period_times, t_max, dt, alpha_array, sq_wave, go_wave):
    """
    Runs the genetic algorithm based on the provided configuration.
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