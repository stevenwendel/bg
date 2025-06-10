"""Run one random chromosome (experimental + control) and print spike counts + scores.

• Run this file directly: `python main.py` or ▶ in VS Code.  
• When `src.__init__` auto‑imports *src.main* during package bootstrap, the
  code **does not execute**, because everything is inside the `if __name__ ==
  "__main__"` guard.  This avoids the double‑run you observed.
"""
from __future__ import annotations

import numpy as np
import src.neuron as neuron  # live module handle (allocates global arrays)
from src.constants import NEURON_NAMES
from src.genetic_algorithm import create_dna, decode_dna_to_matrix
from src.network import create_experiment, run_network
from src.validation import evaluate_conditions

# ---------------------------------------------------------------------
# implementation wrapped in a function to prevent accidental re‑runs
# ---------------------------------------------------------------------

def _run_once():
    # 1. random chromosome → weight matrix ---------------------------
    W = decode_dna_to_matrix(create_dna((0, 400)))

    # 2. build experiment artefacts ----------------------------------
    _, input_waves, alpha_kernel = create_experiment()
    cue_wave, go_wave = input_waves

    # 3. run experimental & control ---------------------------------
    neurons = neuron.create_neurons()
    results: dict[str, int] = {}
    for lbl, ctl in (("experimental", False), ("control", True)):
        neuron.prepare_neurons(neurons, cue_wave, go_wave, ctl)
        neuron.t_pointer = 0
        run_network(neurons, W, alpha_kernel)
        results[lbl] = evaluate_conditions(neuron._SPIKES)[lbl]

    # 4. report ------------------------------------------------------
    print("\nSpike counts (total):")
    for n, c in zip(NEURON_NAMES, neuron._SPIKES.sum(axis=1)):
        print(f"{n:8s}: {int(c)}")

    print("\nScores:")
    print(f"  experimental: {results['experimental']}")
    print(f"  control     : {results['control']}")
    print(f"  TOTAL       : {results['experimental'] + results['control']}\n")


# ---------------------------------------------------------------------
# only execute when run as a script, not when imported indirectly
# ---------------------------------------------------------------------
if __name__ == "__main__":
    _run_once()
