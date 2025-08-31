"""Entry‑point: run one random network instance and print spike counts + scores.

Usage
-----
Just hit ▶ Run in VS Code or do:

    python main.py

Flags aren’t required—the script always builds a fresh random chromosome and
executes one experimental + control pass.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

from src.constants import NEURON_NAMES, TMAX
from src.genetic_algorithm import create_dna, decode_dna_to_matrix
from src.network import create_experiment, run_network
from src.neuron import create_neurons, prepare_neurons, _SPIKES
from src.validation import evaluate_conditions

# ---------------------------------------------------------------------
# 1.  Build random weight matrix
# ---------------------------------------------------------------------
DNA_BOUNDS = (0, 400)          # tweak if you want
random_dna = create_dna(DNA_BOUNDS)       # (len(ACTIVE_SYNAPSES),)
W = decode_dna_to_matrix(random_dna)      # (N, N)

# ---------------------------------------------------------------------
# 2.  Build experiment artefacts
# ---------------------------------------------------------------------
_, input_waves, alpha_kernel = create_experiment()
cue_wave, go_wave = input_waves

# ---------------------------------------------------------------------
# 3.  Initialise neurons & run network (experimental + control)
# ---------------------------------------------------------------------
neurons = create_neurons()
all_scores: dict[str, int] = {}

for label, is_control in (("experimental", False), ("control", True)):
    prepare_neurons(neurons, cue_wave, go_wave, is_control)
    # run network (updates _SPIKES inside module)
    run_network(neurons, W, alpha_kernel)
    all_scores[label] = evaluate_conditions(_SPIKES)[label]

# ---------------------------------------------------------------------
# 4.  Print summary
# ---------------------------------------------------------------------
print("\nSpike counts (TOTAL) per neuron:\n")
counts = _SPIKES.sum(axis=1)
for n, c in zip(NEURON_NAMES, counts):
    print(f"{n:8s}: {int(c)}")

total = all_scores["experimental"] + all_scores["control"]
print("\nScores:")
print(f"  experimental: {all_scores['experimental']}")
print(f"  control     : {all_scores['control']}")
print(f"  TOTAL       : {total}")
