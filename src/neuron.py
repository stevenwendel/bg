import numpy as np

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

    def __str__(self):
        return f"Voltage is set to {self.V} and recovery to {self.u}"

    def restart(self):
        self.V = self.vr
        self.u = 0.0

    def update(self, dt, I_ext=0, sigma=0):
        noise = np.random.normal(0, 10)
        dV = (self.k * (self.V - self.vr) * (self.V - self.vt) - self.u + I_ext + self.E + sigma * noise) / self.C
        du = self.a * (self.b * (self.V - self.vr) - self.u)

        self.V += dV * dt
        self.u += du * dt

        if self.V >= self.vpeak:
            self.V = self.vreset
            self.u += self.d
            self.spiked = True
            return self.V, self.u, self.spiked

        self.spiked = False
        return self.V, self.u, self.spiked