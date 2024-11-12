import unittest
import numpy as np
import pandas as pd
from validation import is_firing, score_neuron, score_run, define_criteria
from src.neuron import Izhikevich

class TestValidation(unittest.TestCase):
    def setUp(self):
        # Initialize neurons
        self.neuron1 = Izhikevich(name='Neuron1', neuron_type='rs')
        self.neuron2 = Izhikevich(name='Neuron2', neuron_type='msn')
        self.neurons = [self.neuron1, self.neuron2]
        
        # Assigning spike times
        self.neuron1.spike_times = np.array([0,1,0,1,0,1,0,1,0,1])
        self.neuron2.spike_times = np.array([1,0,1,0,1,0,1,0,1,0])
        
        # Define criteria
        self.experiment_criteria = pd.DataFrame([
            [1,0,0],
            [1,1,0]
        ], index=['Neuron1', 'Neuron2'], columns=['s1', 's2', 's3'])
        
        self.control_criteria = pd.DataFrame([
            [1,0,0],
            [0,0,0]
        ], index=['Neuron1', 'Neuron2'], columns=['s1', 's2', 's3'])
        
        self.validation_thresholds = [0.3,0.3]  # Example thresholds
        self.validation_period_times = [[0,5], [5,10]]
    
    def test_validate_true(self):
        result = is_firing(self.neuron1, [0,10], 0.4, True)
        self.assertTrue(result)
    
    def test_validate_false(self):
        result = is_firing(self.neuron2, [0,10], 0.4, True)
        self.assertFalse(result)
    
    def test_score_neuron(self):
        score, _ = score_neuron(self.neuron1, self.experiment_criteria, self.validation_thresholds, self.validation_period_times)
        self.assertEqual(score, 2)
    
    def test_score_run(self):
        total_score = score_run(self.neurons, self.experiment_criteria, self.validation_thresholds, self.validation_period_times)
        self.assertEqual(total_score, 2)
    
    def test_define_criteria(self):
        epochs = {
            'sample': [0,1000],
            'delay': [1000,2000]
        }
        number_of_subdivisions = 2
        my_free_weights_names = ['A_B', 'B_C']
        experiment_criteria, control_criteria, validation_period_times, validation_period_names = define_criteria(
            epochs, number_of_subdivisions, self.neurons, my_free_weights_names
        )
        # Check shapes
        self.assertEqual(experiment_criteria.shape, (2,4))
        self.assertEqual(control_criteria.shape, (2,4))
        self.assertEqual(len(validation_period_times),4)
        self.assertEqual(len(validation_period_names),4)

if __name__ == '__main__':
    unittest.main()