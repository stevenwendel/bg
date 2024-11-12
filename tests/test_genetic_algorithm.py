import unittest
import numpy as np
import pandas as pd
from src.genetic_algorithm import initialize_genetic_algorithm, run_genetic_algorithm
from src.neuron import Izhikevich
from src.utils import load_dna
from src.network import run_network

class TestGeneticAlgorithm(unittest.TestCase):
    def setUp(self):
        # Define dummy free weights and path
        self.my_free_weights = [["A", "B"], ["B", "C"], ["C", "A"]]
        self.path = ''
        
        # Define dummy neurons
        self.neurons = [
            Izhikevich(name='A', neuron_type='rs'),
            Izhikevich(name='B', neuron_type='msn'),
            Izhikevich(name='C', neuron_type='rs')
        ]
        
        # Define dummy criteria
        self.experiment_criteria = pd.DataFrame([
            [1,0,0],
            [1,1,0],
            [0,0,0]
        ], index=['A', 'B', 'C'], columns=['s1', 's2', 's3'])
        
        self.control_criteria = pd.DataFrame([
            [1,0,0],
            [0,0,0],
            [0,0,0]
        ], index=['A', 'B', 'C'], columns=['s1', 's2', 's3'])
        
        self.validation_thresholds = [10, 10, 10]
        self.validation_period_times = [[0,100], [100,200], [200,300]]
    
    def test_initialize_genetic_algorithm(self):
        config = initialize_genetic_algorithm(
            pop_size=10,
            mut_rate=0.1,
            mut_sigma=0.3,
            rank_depth=3,
            crossover_point=1,
            num_gen=5,
            elite_passthrough=2,
            bounds=[0, 100],
            my_free_weights=self.my_free_weights,
            path=self.path
        )
        self.assertEqual(len(config['dnas']), 10)
        self.assertEqual(config['mutation_rate'], 0.1)
        self.assertEqual(config['mutation_sigma'], 0.3)
    
    def test_spawn_next_generation(self):
        # Initialize dummy population
        current_population = [
            ([10, 20, 30], 100),
            ([20, 30, 40], 200),
            ([30, 40, 50], 300),
            ([40, 50, 60], 400),
            ([50, 60, 70], 500)
        ]
        rank_depth = 3
        mut_rate = 0.1
        mut_sigma = 0.3
        elite_passthrough = 2
        pop_size = 5
        new_dnas = genetic_algorithm.spawn_next_generation(
            current_population, rank_depth, mut_rate, mut_sigma,
            elite_passthrough, pop_size, self.my_free_weights
        )
        self.assertEqual(len(new_dnas), pop_size)
    
    def test_score_population(self):
        # Create dummy DNA
        dnas = [[10, 20, 30], [20, 30, 40]]
        
        # Define a dummy run_network function that assigns a dummy score
        def dummy_run_network(weight_matrix, neurons, sq_wave, go_wave, t_max, dt, alpha_array, control=False):
            pass  # Do nothing
        
        # Monkey patch run_network
        original_run_network = run_network
        try:
            globals()['run_network'] = dummy_run_network
            
            # Define a dummy score_run function
            def dummy_score_run(validation_neurons, criteria, thresholds, periods):
                return 100  # Return a constant score
            
            # Monkey patch score_run
            original_score_run = genetic_algorithm.score_run
            globals()['genetic_algorithm.score_run'] = dummy_score_run
            
            scores = genetic_algorithm.score_population(
                dnas, self.my_free_weights, len(dnas),
                self.neurons, self.experiment_criteria, self.control_criteria
            )
            
            for score in scores:
                self.assertEqual(score[1], 200)  # Each DNA gets 100 + 100
        finally:
            globals()['run_network'] = original_run_network
            globals()['genetic_algorithm.score_run'] = original_score_run

if __name__ == '__main__':
    unittest.main()