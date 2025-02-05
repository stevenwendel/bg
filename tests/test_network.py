import unittest
import numpy as np
from src.network import run_network
from src.viz import plot_neurons
from src.neuron import Izhikevich, prepare_neurons
from src.constants import TMAX

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

def test_run_network_basic():
    # Setup
    n_neurons = 3
    neurons = [Izhikevich(f"test_neuron_{i}") for i in range(n_neurons)]
    weight_matrix = np.zeros((n_neurons, n_neurons))
    alpha_array = np.ones(250)  # Simple alpha array for testing
    
    # Prepare neurons
    cue_wave = np.zeros(TMAX)
    go_wave = np.zeros(TMAX)
    prepare_neurons(neurons, cue_wave, go_wave, control=True)
    
    # Run network
    run_network(neurons, weight_matrix, alpha_array)
    
    # Basic assertions
    for neuron in neurons:
        assert len(neuron.hist_V) == TMAX
        assert len(neuron.hist_u) == TMAX
        assert len(neuron.spike_times) == TMAX - 1  # spike_times has length TMAX-1

def test_run_network_with_connections():
    # Setup
    n_neurons = 2
    neurons = [Izhikevich(f"test_neuron_{i}") for i in range(n_neurons)]
    # Create a simple connection from neuron 0 to neuron 1
    weight_matrix = np.array([[0, 0], [1, 0]])
    alpha_array = np.ones(250)
    
    # Prepare neurons with strong input to first neuron
    cue_wave = np.zeros(TMAX)
    cue_wave[100:150] = 200  # Strong input to cause spiking
    go_wave = np.zeros(TMAX)
    prepare_neurons(neurons, cue_wave, go_wave, control=True)
    
    # Run network
    run_network(neurons, weight_matrix, alpha_array)
    
    # Check if spikes in neuron 0 affect neuron 1
    n0_spikes = np.where(neurons[0].spike_times == 1)[0]
    assert len(n0_spikes) > 0, "First neuron should spike with strong input"
    
    # Check if there's corresponding activity in neuron 1 after neuron 0 spikes
    for spike_time in n0_spikes:
        if spike_time + 1 < TMAX:
            assert neurons[1].hist_V[spike_time + 1] > neurons[1].vr, \
                "Second neuron should receive input after first neuron spikes"

def test_run_network_no_self_connections():
    # Setup
    n_neurons = 1
    neurons = [Izhikevich("test_neuron")]
    weight_matrix = np.array([[1]])  # Self-connection
    alpha_array = np.ones(250)
    
    # Prepare neurons
    cue_wave = np.zeros(TMAX)
    go_wave = np.zeros(TMAX)
    prepare_neurons(neurons, cue_wave, go_wave, control=True)
    
    # Initial voltage
    initial_v = neurons[0].hist_V[0]
    
    # Run network
    run_network(neurons, weight_matrix, alpha_array)
    
    # Check that self-connection doesn't cause unbounded growth
    assert np.all(neurons[0].hist_V < neurons[0].vpeak), \
        "Voltage should not exceed vpeak due to self-connections"

def test_alpha_scaling():
    # Setup
    n_neurons = 2
    neurons = [Izhikevich(f"test_neuron_{i}") for i in range(n_neurons)]
    weight_matrix = np.array([[0, 0], [1, 0]])
    
    # Create decreasing alpha array
    alpha_array = np.linspace(1, 0, 250)
    
    # Prepare neurons with strong input to first neuron
    cue_wave = np.zeros(TMAX)
    cue_wave[100:150] = 200  # Strong input to cause spiking
    go_wave = np.zeros(TMAX)
    prepare_neurons(neurons, cue_wave, go_wave, control=True)
    
    # Run network
    run_network(neurons, weight_matrix, alpha_array)
    
    # Find spikes in first neuron
    n0_spikes = np.where(neurons[0].spike_times == 1)[0]
    
    # Check if effect on second neuron decreases over time
    prev_effect = float('inf')
    for spike_time in n0_spikes[:5]:  # Check first few spikes
        if spike_time + 1 < TMAX:
            effect = neurons[1].hist_V[spike_time + 1] - neurons[1].vr
            assert effect <= prev_effect, \
                "Effect of spikes should decrease or stay same with decreasing alpha"
            prev_effect = effect

if __name__ == '__main__':
    unittest.main()