import unittest
import numpy as np
import pandas as pd
from random import *
from src.genetic_algorithm import *
from src.genetic_algorithm import initialize_genetic_algorithm, run_genetic_algorithm
from src.neuron import Izhikevich
from src.utils import load_dna
from src.network import run_network

def test_create_and_mutate():
    # Create single DNA
    dna = create_dna([0,10])

    # Mutate all genes in DNA
    for i, gene in enumerate(dna):
        dna[i] = random.normalvariate(gene,MUT_SIGMA)

def test_spawn_next_generation():

    # Create POP_SIZE 
    curr_population=[create_dna(DNA_BOUNDS) for _ in range(POP_SIZE)]
    population_results = []

    for i, curr_dna in enumerate(curr_population):
        dna_score  = random.randint(40,80)

        population_results.append({
            'dna': curr_dna,
            'dna_score' : dna_score
        })

    print(population_results)

    new_population = spawn_next_population(population_results)

if __name__ == '__main__':
    unittest.main()