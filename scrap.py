import sys, os

# Add the src directory to sys.path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.append(src_path)

import numpy as np
import pandas as pd
import shelve
from src.neuron import *
from src.utils import *
from src.constants import * 
from src.network import *
from src.validation import *
from src.viz import *
from src.genetic_algorithm import *
from copy import copy
from datetime import datetime



# with shelve.open('ga_database', 'r') as shelf:
#     for key in shelf.keys():
#         print(repr(key), repr(shelf[key]))


# # Create single DNA
# dna = create_dna([0,10])

# # Mutate all genes in DNA
# for i, gene in enumerate(dna):
#     dna[i] = random.normalvariate(gene,MUT_SIGMA)

# # Create POP_SIZE 
# curr_population=[create_dna(DNA_BOUNDS) for _ in range(POP_SIZE)]
# population_results = []

# for i, curr_dna in enumerate(curr_population):
#     dna_score  = random.randint(40,80)

#     population_results.append({
#         'dna': curr_dna,
#         'dna_score' : dna_score
#     })

# print(population_results)

# new_population = spawn_next_population(population_results)

gene = 300
gene = random.normalvariate(gene, gene*MUT_SIGMA)
print(gene)














"""
# with open('./data/experimental_neurons.pkl', 'rb') as f:
#     experimental_neurons = pickle.load(f)

# with open('./data/control_neurons.pkl', 'rb') as f:
#     control_neurons = pickle.load(f)

# assert len(experimental_neurons) == len(control_neurons)

# binned_differences = get_binned_differences(experimental_neurons, control_neurons, 100)
# plot_binned_differences(binned_differences, 100)

# Convert lists to NumPy arrays
experiment_criteria_by_epoch = pd.Series({
    "Somat": [0, 1, 0, 0, 0],
    "ALMprep": [0, 1, 1, 0, 0],
    "ALMinter": [0, 0, 0, 0, 0],
    "ALMresp": [0, 0, 1, 0, 0],
    "SNR1": [0, 1, 1, 0, 0],
    "SNR2": [0, 0, 0, 1, 1],
    "VMprep": [1, 1, 0, 0, 0],
    "VMresp": [0, 0, 0, 1, 1],
    "PPN": [0, 0, 0, 0, 0]
})

# Define criteria by epoch for control
control_criteria_by_epoch = pd.Series({
    "Somat": [0, 0, 0, 0, 0],
    "ALMprep": [0, 0, 0, 0, 0],
    "ALMinter": [0, 0, 0, 1, 0],
    "ALMresp": [0, 0, 0, 0, 0],
    "SNR1": [1, 1, 1, 1, 1],
    "SNR2": [1, 1, 1, 1, 1],
    "VMprep": [0, 0, 0, 0, 0],
    "VMresp": [0, 0, 0, 0, 0],
    "PPN": [0, 0, 0, 0, 0]
})

# Element-wise subtraction
difference_criteria_by_epoch = experiment_criteria_by_epoch.combine(
    control_criteria_by_epoch, 
    lambda x, y: np.subtract(x, y)
)

print(difference_criteria_by_epoch)

"""