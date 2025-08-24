"""Genetic‑algorithm helpers for the BG network.

Changes (2025‑06‑10)
--------------------
* **Uniform crossover** (per‑gene swap‑probability `swap_p`).
* **Self‑adaptive Gaussian mutation** – each individual carries its own
  sigma; bad step‑sizes die out automatically.
* Population spawner uses tournament selection + niching to keep diversity.
* DNA always stored as **int32** for readability; converted to float32 when
  building the weight matrix.
"""
from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np

from src.constants import ACTIVE_SYNAPSES, INHIBITORY_NEURONS, NEURON_NAMES, CRITICAL_CONNECTIONS

# ------------------------------------------------------------------
# Pre‑computed index maps
# ------------------------------------------------------------------
_ORIGIN_IDX = np.array([NEURON_NAMES.index(o) for o, _ in ACTIVE_SYNAPSES], dtype=np.int16)
_TARGET_IDX = np.array([NEURON_NAMES.index(t) for _, t in ACTIVE_SYNAPSES], dtype=np.int16)
_N_NEURON   = len(NEURON_NAMES)
_INHIB_MASK  = np.isin(_ORIGIN_IDX, [NEURON_NAMES.index(n) for n in INHIBITORY_NEURONS])

# Pre-compute critical connection indices for fast constraint enforcement
_CRITICAL_INDICES = {}
for (origin, target), constraint in CRITICAL_CONNECTIONS.items():
    try:
        origin_idx = NEURON_NAMES.index(origin)
        target_idx = NEURON_NAMES.index(target)
        # Find the DNA index for this connection
        for i, (o, t) in enumerate(ACTIVE_SYNAPSES):
            if o == origin and t == target:
                _CRITICAL_INDICES[i] = constraint
                break
    except (ValueError, IndexError):
        print(f"Warning: Critical connection {origin}->{target} not found in ACTIVE_SYNAPSES")

__all__ = [
    "create_dna",
    "decode_dna_to_matrix", 
    "enforce_critical_constraints",
    "uniform_crossover",
    "mutate_gauss",
    "spawn_next_population",
]

# ------------------------------------------------------------------
# 1.  DNA creation / decode
# ------------------------------------------------------------------

def create_dna(bounds: Tuple[int, int]) -> np.ndarray:
    low, high = bounds
    dna = np.random.randint(low, high + 1, size=_ORIGIN_IDX.size, dtype=np.int32)
    dna[_INHIB_MASK] *= -1
    
    # Enforce critical connection constraints for initial population
    dna = enforce_critical_constraints(dna, bounds)
    
    return dna


def decode_dna_to_matrix(dna: np.ndarray) -> np.ndarray:
    W = np.zeros((_N_NEURON, _N_NEURON), np.float32)
    np.add.at(W, (_ORIGIN_IDX, _TARGET_IDX), dna.astype(np.float32))
    return W


def enforce_critical_constraints(dna: np.ndarray, bounds: Tuple[int, int]) -> np.ndarray:
    """
    Enforce minimum absolute weight thresholds on critical connections.
    
    Args:
        dna: DNA vector to constrain
        bounds: (low, high) bounds for DNA values
        
    Returns:
        DNA vector with critical connections above minimum thresholds
    """
    constrained_dna = dna.copy()
    low, high = bounds
    
    for dna_idx, constraint in _CRITICAL_INDICES.items():
        min_abs = constraint["min_abs"]
        current_val = constrained_dna[dna_idx]
        
        if abs(current_val) < min_abs:
            # Determine sign - use negative for inhibitory connections
            is_inhibitory = _INHIB_MASK[dna_idx] if dna_idx < len(_INHIB_MASK) else False
            
            if is_inhibitory:
                # For inhibitory connections, ensure negative and above min_abs magnitude
                constrained_dna[dna_idx] = max(-high, -min_abs)
            else:
                # For excitatory connections, ensure positive and above min_abs magnitude  
                constrained_dna[dna_idx] = min(high, min_abs)
    
    return constrained_dna

# ------------------------------------------------------------------
# 2.  Operators
# ------------------------------------------------------------------

def uniform_crossover(p1: np.ndarray, p2: np.ndarray, swap_p: float = 0.5, bounds: Tuple[int, int] = (0, 500)) -> np.ndarray:
    """Per‑gene uniform crossover with critical connection constraints."""
    mask = np.random.rand(p1.size) < swap_p
    child = np.where(mask, p1, p2)
    child = child.astype(np.int32)
    
    # Enforce critical connection constraints after crossover
    child = enforce_critical_constraints(child, bounds)
    
    return child


def mutate_gauss(dna: np.ndarray, sigma: float, bounds: Tuple[int, int]) -> np.ndarray:
    """
    Gaussian mutation: each element is added to the product of itself and a random number 
    from a Gaussian distribution (mean=0, standard deviation=sigma).
    
    Args:
        dna: DNA vector to mutate
        sigma: Standard deviation for Gaussian noise
        bounds: (low, high) bounds for clipping
    
    Returns:
        Mutated DNA vector
    """
    # Convert to float32 for calculations
    dna_float = dna.astype(np.float32)
    
    # Generate random Gaussian noise for each element
    noise = np.random.normal(0, sigma, size=dna.size)
    
    # Apply mutation: dna_element + (dna_element * gaussian_noise)
    mutated = dna_float + (dna_float * noise)
    
    # Clip to bounds and round
    low, high = bounds
    mutated = np.clip(np.round(mutated), -high, high)
    
    # Fix signs based on neuron type
    mutated[_INHIB_MASK] = -np.abs(mutated[_INHIB_MASK])  # Inhibitory neurons: negative
    mutated[~_INHIB_MASK] = np.abs(mutated[~_INHIB_MASK])  # Excitatory neurons: positive
    
    # Convert to int32 before applying constraints
    mutated = mutated.astype(np.int32)
    
    # Enforce critical connection constraints after mutation
    mutated = enforce_critical_constraints(mutated, bounds)
    
    return mutated

# ------------------------------------------------------------------
# 3.  Population spawning
# ------------------------------------------------------------------

def _tournament(pop: List[dict], k: int) -> np.ndarray:
    """Return winner DNA from k‑sized tournament (higher score wins)."""
    contenders = random.sample(pop, k)
    return max(contenders, key=lambda r: r["dna_score"]) ["dna"]


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


def spawn_next_population(pop_records: List[dict], cfg: dict, generation: int) -> List[np.ndarray]:
    pop_size   = cfg["POP_SIZE"]
    bounds     = tuple(cfg["DNA_BOUNDS"])
    elite_n    = cfg["ELITE_SIZE"]
    rank_depth = cfg["RANK_DEPTH"]
    sigma      = cfg["MUT_SIGMA"]
    mut_rate   = cfg["MUT_RATE"]

    pop_records.sort(key=lambda r: r["dna_score"], reverse=True)
    elites = [r["dna"] for r in pop_records[:elite_n]]
    next_pop = elites.copy()

    # keep‑distance niching threshold (5 % of chromosome length)
    niche_thresh = 0.01 * _ORIGIN_IDX.size

    while len(next_pop) < pop_size:
        p1 = _tournament(pop_records[:rank_depth], 3)
        p2 = _tournament(pop_records[:rank_depth], 3)
        child = uniform_crossover(p1, p2, swap_p=0.5)
        if np.random.rand() < mut_rate:
            child = mutate_gauss(child, sigma, bounds)

        if all(_hamming(child, dna) > niche_thresh for dna in next_pop):
            next_pop.append(child)

    return next_pop
