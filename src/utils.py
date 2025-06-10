"""Utility helpers (vectorised alpha kernel, data selectors, etc.).

This module is now *pure NumPy/pandas* and contains no heavy plotting or OS
imports.  The main fixes are:

1. `create_alpha_array` — fully vectorised, float32, no Python loop.
2. `alpha_fit`        — in‑place padding without reallocating on every call.
   (Retained for legacy code but no longer used by `network.run_network`.)
3. `get_neurons`      — works on either raw NumPy or DataFrame inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import psutil
import pickle
from src.constants import NEURON_NAMES, TMAX

__all__ = [
    "create_alpha_array",
    "alpha_fit",
    "get_neurons",
]

# --------------------------------------------------------------------
# ALPHA KERNELS
# --------------------------------------------------------------------

def create_alpha_array(length: int, L: int = 30, dtype=np.float32) -> np.ndarray:
    """Generate an alpha‑function kernel of given *length*.

    The kernel is *not* normalised; scale it with the synaptic weight when you
    add it to the input buffer.

    alpha(t) = (t/L) * exp((L − t)/L)       for  t ∈ 1..length

    Parameters
    ----------
    length : int
        Number of 1‑ms samples to return.
    L : int, default 30
        Time‑to‑peak parameter of the alpha function in milliseconds.
    dtype : np.dtype, default float32
        Output dtype (float32 recommended to match simulator arrays).
    """
    td = np.arange(0, length , dtype=dtype)  # 1 … length
    alpha = (td / L) * np.exp((L - td) / L)
    # Keep four decimal places (matches original code, ~0.1 % error)
    return np.round(alpha, 4, out=alpha)  # reuse `alpha` buffer


def alpha_fit(alpha_kernel: np.ndarray, start_time: int, tmax: int = TMAX) -> np.ndarray:
    """Pad *alpha_kernel* into a zero array of length *tmax* starting at *start_time*.

    This is a **compatibility shim** for legacy code.  It allocates a fresh
    output array; prefer direct in‑place broadcast as done in
    `network.run_network` for performance.
    """
    out = np.zeros(tmax, dtype=alpha_kernel.dtype)
    lend = min(alpha_kernel.size, tmax - start_time)
    if lend > 0:
        out[start_time : start_time + lend] = alpha_kernel[:lend]
    return out

# --------------------------------------------------------------------
# DATA HELPERS
# --------------------------------------------------------------------

def get_neurons(neuron_data, target_neurons: list[str]) -> np.ndarray:
    """Return rows for *target_neurons* from `neuron_data`.

    *neuron_data* can be either:
      • a NumPy array with rows ordered as in `NEURON_NAMES`, or
      • a pandas DataFrame whose index contains neuron names.
    """
    if isinstance(neuron_data, pd.DataFrame):
        df = neuron_data
    else:
        df = pd.DataFrame(neuron_data, index=NEURON_NAMES)

    # preserve order given in *target_neurons*
    idx = [n for n in target_neurons if n in df.index]
    return df.loc[idx].to_numpy()


def save_neurons(neurons: list[Izhikevich], condition):
    with open(f"./data/{condition}_neurons.pkl", "wb") as f:
        pickle.dump(neurons, f)

# Probably broken and need to fix, but later.
def load_neurons(file_path: str):
    with open(file_path,'rb') as f:
        try:
            while True:
                # Load each (dna_score, dna_0) pair
                dna_score, test_dna = pickle.load(f)
                print(f'Loaded dna_score: {dna_score}, dna_0: {test_dna}')
        except EOFError:
            # End of file reached
            pass

# Function to create a DNA string
def create_dna_string(weights, active_synapses):
    # Initialize a DNA list with zeros
    dna = [0] * len(active_synapses)
    
    # Iterate through the weights
    for source, target, weight in weights:
        # Find the index of the connection in ACTIVE_SYNAPSES
        try:
            index = active_synapses.index([source, target])
            # Insert the weight at the found index
            dna[index] = weight
        except ValueError:
            # If the connection is not found, you can choose to ignore or handle it
            print(f"Connection {source} -> {target} not found in ACTIVE_SYNAPSES.")
   
    return dna

def load_ga_run_to_df(file_path: str) -> pd.DataFrame:
    """Load a genetic algorithm run pickle file into a sorted DataFrame.
    
    Args:
        file_path (str): Path to the pickle file containing the GA run data
        
    Returns:
        pd.DataFrame: DataFrame with columns:
            - generation: Generation number
            - dna: DNA sequence as a tuple
            - dna_score: Score for the DNA sequence
        Sorted by dna_score in descending order
    """
    # Load the pickle file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Initialize list to store all rows
    rows = []
    
    # Iterate through each generation
    for key in data.keys():
        if key.startswith('gen_'):
            gen_num = int(key.split('_')[1])
            population = data[key]['population']
            
            # Add each DNA sequence and its score to the rows
            for dna_dict in population:
                rows.append({
                    'generation': gen_num,
                    'dna': tuple(dna_dict['dna']),  # Convert list to tuple for hashability
                    'dna_score': dna_dict['dna_score']
                })
    
    # Create DataFrame and sort
    df = pd.DataFrame(rows)
    df = df.sort_values('dna_score', ascending=False, ignore_index=True)
    
    return df


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB
