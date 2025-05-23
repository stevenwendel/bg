import numpy as np
from src.constants import *

class Izhikevich:
    """A simple Izhikevich neuron.
    Default parameterization gives a Regular Spiking neuron.
    """
    def __init__(self, name='generic_rs', neuron_type="rs"):
        parameters = {
            'rs': [0.03, -2.0, -50.0, 100.0, 0.7, -60.0, -40.0, 35.0, 0.0, -60.0, 0.0, 100.0],
            'msn': [0.01, -20.0, -55.0, 150.0, 1.0, -80.0, -25.0, 40.0, 70.0, -60.0, 0.0, 50.0]
        }

        self.name = name
        self.a, self.b, self.vreset, self.d, self.k, self.vr, self.vt, self.vpeak, self.E, self.V, self.u, self.C = parameters[neuron_type]
        self.spiked = False
        self.input = None
        self.spike_times = None
        self.hist_V = None
        self.hist_u = None
        self._initialized = False

    def initialize_arrays(self, tmax):
        """Initialize arrays only when needed and with exact size"""
        if not self._initialized:
            self.input = np.zeros(tmax, dtype=np.float32)  # Use float32 instead of float64
            self.spike_times = np.zeros(tmax, dtype=np.int8)  # Use int8 for binary spike data
            self.hist_V = np.zeros(tmax, dtype=np.float32)
            self.hist_u = np.zeros(tmax, dtype=np.float32)
            self._initialized = True

    def cleanup(self):
        """Clear arrays to free memory"""
        self.input = None
        self.spike_times = None
        self.hist_V = None
        self.hist_u = None
        self._initialized = False

    def __str__(self):
        return f"Voltage is set to {self.V} and recovery to {self.u}"

    def reset(self):
        """Reset neuron state and clear arrays"""
        self.V = self.vr
        self.u = 0.0
        self.cleanup()

    def update(self, I_ext=0, sigma=0): 
        noise = np.random.normal(0, sigma)

        dV = (self.k * (self.V - self.vr) * (self.V - self.vt) - self.u + I_ext + self.E + sigma * noise) / self.C
        du = self.a * (self.b * (self.V - self.vr) - self.u)

        self.V += dV    # Note: dt=1
        self.u += du    # Note: dt=1


        if self.V >= self.vpeak:
            self.V = self.vreset
            self.u += self.d
            self.spiked = True
        else:
            self.spiked = False

        return self.V, self.u, self.spiked

    

def create_neurons() ->list[Izhikevich]:

    # Instantiating neurons
    neurons = []
    for neu in NEURON_NAMES:
        if neu in ["MSN1", "MSN2", "MSN3"]:
            neuron_instance = Izhikevich(name = neu, neuron_type="msn")
        else:
            neuron_instance = Izhikevich(name = neu, neuron_type="rs")

        if neuron_instance.name in ["SNR1", "SNR2", "SNR3"]:
            neuron_instance.E = 120.0 
        if neuron_instance.name == "PPN":
            neuron_instance.E = 100.0

        neurons.append(neuron_instance) # Creates a list of all Iz neurons; note, these are the actual objects, not a list of names!

    return neurons

def prepare_neurons(neurons: list[Izhikevich], cue_wave, go_wave, control):
    for neu in neurons:
        neu.reset() 
        neu.initialize_arrays(TMAX)
        neu.hist_V[0] = neu.V
        neu.hist_u[0] = neu.u

        if neu.name == "Somat" and not control:
            neu.input += cue_wave
        if neu.name == "PPN":
            neu.input += go_wave
