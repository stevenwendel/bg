import src.neuron as neuron
from src.network import run_network
from src.validation import evaluate_conditions
from src.neuron import create_neurons, prepare_neurons, _SPIKES

def score_dna(dna_matrix, input_waves, alpha_kernel):
    """Return TOTAL score for one chromosome."""
    cue_wave, go_wave = input_waves
    neurons = create_neurons()

    total = 0
    for label, ctl in (("experimental", False), ("control", True)):
        prepare_neurons(neurons, cue_wave, go_wave, ctl)
        neuron.t_pointer = 0
        run_network(neurons, dna_matrix, alpha_kernel)
        total += evaluate_conditions(_SPIKES)[label]
    return total