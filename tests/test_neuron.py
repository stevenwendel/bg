import unittest
import numpy as np
from src.neuron import Izhikevich

class TestIzhikevichNeuron(unittest.TestCase):
    def setUp(self):
        self.rs_neuron = Izhikevich(name='TestRS', neuron_type='rs')
        self.msn_neuron = Izhikevich(name='TestMSN', neuron_type='msn')
    
    def test_initialization_rs(self):
        self.assertEqual(self.rs_neuron.name, 'TestRS')
        self.assertEqual(self.rs_neuron.V, self.rs_neuron.vr)
        self.assertEqual(self.rs_neuron.u, 0.0)
        self.assertFalse(self.rs_neuron.spiked)
    
    def test_initialization_msn(self):
        self.assertEqual(self.msn_neuron.name, 'TestMSN')
        self.assertEqual(self.msn_neuron.V, self.msn_neuron.vr)
        self.assertEqual(self.msn_neuron.u, 0.0)
        self.assertFalse(self.msn_neuron.spiked)
    
    def test_restart(self):
        self.rs_neuron.V = 0.0
        self.rs_neuron.u = 10.0
        self.rs_neuron.restart()
        self.assertEqual(self.rs_neuron.V, self.rs_neuron.vr)
        self.assertEqual(self.rs_neuron.u, 0.0)
    
    def test_update_no_spike(self):
        self.rs_neuron.V = self.rs_neuron.vr
        self.rs_neuron.u = 0.0
        V, u, spike = self.rs_neuron.update(dt=1, I_ext=0, sigma=0)
        self.assertFalse(spike)
        self.assertLess(V, self.rs_neuron.vpeak)
    
    def test_update_with_spike(self):
        self.rs_neuron.V = self.rs_neuron.vpeak - 1
        self.rs_neuron.u = 0.0
        V, u, spike = self.rs_neuron.update(dt=1, I_ext=self.rs_neuron.vpeak, sigma=0)
        self.assertTrue(spike)
        self.assertEqual(V, self.rs_neuron.vreset)
        self.assertEqual(u, self.rs_neuron.d)
    
    def test_spike_times_collection(self):
        self.rs_neuron.spike_times = np.zeros(10)
        self.rs_neuron.V = self.rs_neuron.vpeak - 1
        self.rs_neuron.u = 0.0
        _, _, spike = self.rs_neuron.update(dt=1, I_ext=self.rs_neuron.vpeak, sigma=0)
        self.rs_neuron.spike_times[0] = spike
        self.assertEqual(self.rs_neuron.spike_times[0], 1)

if __name__ == '__main__':
    unittest.main()