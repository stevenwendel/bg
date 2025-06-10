"""Smoke‑test: run a fixed weight list through the network **without** criteria.

How to run (project root):
    python -m tests.run_manual_weights
If you double‑click the file or use `python tests/run_manual_weights.py`, the
first two lines below add the repo root to `sys.path` so `import src.*` still
works.
"""
from __future__ import annotations

# --- make `src` importable even when executed as plain script ---------
import sys, pathlib, os
repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import src.neuron as neuron  # one module handle to keep globals consistent
from src.constants import NEURON_NAMES, TMAX
from src.network import create_experiment
from src.validation import evaluate_conditions

# --------------------------------------------------------------------
# Hand‑crafted weights -------------------------------------------------
# --------------------------------------------------------------------
new_jh_weights = [
    ("Somat", "ALMprep", 40),
    ("Somat", "MSN1", 220),
    ("MSN1", "SNR1", -90),
    ("SNR1", "VMprep", -10),
    ("VMprep", "ALMprep", 70),
    ("ALMprep", "VMprep", 80),
    ("ALMprep", "MSN2", 320),
    ("MSN2", "SNR2", -50),
    ("SNR2", "VMresp", -100),
    ("PPN", "THALgo", 60),
    ("THALgo", "ALMinter", 55),
    ("ALMinter", "ALMprep", -50),
    ("THALgo", "ALMresp", 30),
    ("ALMresp", "MSN3", 320),
    ("MSN3", "SNR3", -90),
    ("SNR3", "VMresp", -50),
    ("VMresp", "ALMresp", 85),
    ("ALMresp", "VMresp", 90),
]

N = len(NEURON_NAMES)
W = np.zeros((N, N), dtype=np.float32)
for pre, post, w in new_jh_weights:
    i = NEURON_NAMES.index(pre)
    j = NEURON_NAMES.index(post)
    W[i, j] += w

# --------------------------------------------------------------------
# Build experiment artefacts -----------------------------------------
# --------------------------------------------------------------------
_, input_waves, alpha_kernel = create_experiment()
cue_wave, go_wave = input_waves

# --------------------------------------------------------------------
# Initialise neurons & run network --- --------------------------------
# --------------------------------------------------------------------
neurons = neuron.create_neurons()
neuron.prepare_neurons(neurons, cue_wave, go_wave, control=False)

for t in range(TMAX - 1):
    ext_I = neuron._INPUT[:, t]
    spikes = neuron.vectorised_step(ext_I)
    # simple delta‑current synapse
    # alpha‑shaped PSP (NEW)
    if spikes.any():
        post_current = spikes.astype(np.float32) @ W     # shape (N,)

        # how many future samples still fit before TMAX?
        lend = min(alpha_kernel.size, TMAX - (t + 1))

        # broadcast: (N, lend)  =  (N,1)  *  (lend,)
        neuron._INPUT[:, t + 1 : t + 1 + lend] += post_current[:, None] * alpha_kernel[:lend]
        
# final step (no future syn current)
neuron.vectorised_step(neuron._INPUT[:, TMAX - 1])

# --------------------------------------------------------------------
# Report spike counts -------------------------------------------------
# --------------------------------------------------------------------
counts = neuron._SPIKES.sum(axis=1)
print("\nSpike counts with manual weights:\n")
for name, c in zip(NEURON_NAMES, counts):
    print(f"{name:8s}: {int(c)}")

scores = evaluate_conditions(neuron._SPIKES)   # returns dict
total  = scores['experimental'] + scores['control']

print(f"\nExperimental score: {scores['experimental']}")
print(f"Control score:      {scores['control']}")
print(f"TOTAL:              {total}")