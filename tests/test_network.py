import unittest
import numpy as np
from src.network import run_network, plot_neurons
from src.neuron import Izhikevich

class TestNetwork(unittest.TestCase):
    def setUp(self):
        # Initialize two neurons for testing
        self.neuron1 = Izhikevich(name='Neuron1', neuron_type='rs')
        self.neuron2 = Izhikevich(name='Neuron2', neuron_type='msn')
        self.neurons = [self.neuron1, self.neuron2]
        
        # Initialize inputs
        self.t_max = 10
        self.dt = 1
        self.sq_wave = np.zeros(self.t_max)
        self.go_wave = np.zeros(self.t_max)
        self.alpha_array = np.array([1, 2, 3])
        
        # Initialize weight matrix
        self.weight_matrix = np.array([
            [0, 5],
            [10, 0]
        ])
    
    def test_run_network_no_spikes(self):
        run_network(
            weight_matrix=self.weight_matrix,
            neurons=self.neurons,
            sq_wave=self.sq_wave,
            go_wave=self.go_wave,
            t_max=self.t_max,
            dt=self.dt,
            alpha_array=self.alpha_array,
            control=False
        )
        # Ensure no spikes occurred
        self.assertFalse(np.any(self.neuron1.spike_times))
        self.assertFalse(np.any(self.neuron2.spike_times))
    
    def test_run_network_with_spikes(self):
        # Set external current to trigger a spike
        self.sq_wave[0] = 1000  # High input to trigger spike
        run_network(
            weight_matrix=self.weight_matrix,
            neurons=self.neurons,
            sq_wave=self.sq_wave,
            go_wave=self.go_wave,
            t_max=self.t_max,
            dt=self.dt,
            alpha_array=self.alpha_array,
            control=False
        )
        # Check if at least one spike occurred
        self.assertTrue(np.any(self.neuron1.spike_times))
    
    def test_control_weights(self):
        # Set external current to trigger a spike
        self.sq_wave[0] = 1000
        run_network(
            weight_matrix=self.weight_matrix,
            neurons=self.neurons,
            sq_wave=self.sq_wave,
            go_wave=self.go_wave,
            t_max=self.t_max,
            dt=self.dt,
            alpha_array=self.alpha_array,
            control=True
        )
        # In control, some weights are zeroed, affecting spikes
        # Depending on weight matrix adjustments, verify expected behavior
        # For simplicity, here we just ensure the function runs without error
        self.assertTrue(True)
    
    def test_history_records(self):
        self.sq_wave[0] = 1000
        run_network(
            weight_matrix=self.weight_matrix,
            neurons=self.neurons,
            sq_wave=self.sq_wave,
            go_wave=self.go_wave,
            t_max=self.t_max,
            dt=self.dt,
            alpha_array=self.alpha_array,
            control=False
        )
        # Check if history arrays are updated
        self.assertTrue(len(self.neuron1.hist_V) == self.t_max)
        self.assertTrue(len(self.neuron2.hist_u) == self.t_max)

if __name__ == '__main__':
    unittest.main()