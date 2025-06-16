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

from src.constants import ACTIVE_SYNAPSES, INHIBITORY_NEURONS, NEURON_NAMES

# ------------------------------------------------------------------
# Pre‑computed index maps
# ------------------------------------------------------------------
_ORIGIN_IDX = np.array([NEURON_NAMES.index(o) for o, _ in ACTIVE_SYNAPSES], dtype=np.int16)
_TARGET_IDX = np.array([NEURON_NAMES.index(t) for _, t in ACTIVE_SYNAPSES], dtype=np.int16)
_N_NEURON   = len(NEURON_NAMES)
_INHIB_MASK  = np.isin(_ORIGIN_IDX, [NEURON_NAMES.index(n) for n in INHIBITORY_NEURONS])

__all__ = [
    "create_dna",
    "decode_dna_to_matrix",
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
    return dna


def decode_dna_to_matrix(dna: np.ndarray) -> np.ndarray:
    W = np.zeros((_N_NEURON, _N_NEURON), np.float32)
    np.add.at(W, (_ORIGIN_IDX, _TARGET_IDX), dna.astype(np.float32))
    return W

# ------------------------------------------------------------------
# 2.  Operators
# ------------------------------------------------------------------

def uniform_crossover(p1: np.ndarray, p2: np.ndarray, swap_p: float = 0.5) -> np.ndarray:
    """Per‑gene uniform crossover."""
    mask = np.random.rand(p1.size) < swap_p
    child = np.where(mask, p1, p2)
    return child.astype(np.int32)


def mutate_gauss(dna: np.ndarray, sigma: float, bounds: Tuple[int, int]) -> np.ndarray:
    """Gaussian mutation with automatic rounding and sign fix."""
    dna = dna.astype(np.float32) + np.dot(np.random.normal(0, sigma, size=dna.size), dna.astype(np.float32))
    low, high = bounds
    dna = np.clip(np.round(dna), -high, high)
    dna[_INHIB_MASK] = -np.abs(dna[_INHIB_MASK])
    dna[~_INHIB_MASK] =  np.abs(dna[~_INHIB_MASK])
    return dna.astype(np.int32)

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

    pop_records.sort(key=lambda r: r["dna_score"], reverse=True)
    elites = [r["dna"] for r in pop_records[:elite_n]]
    next_pop = elites.copy()

    # keep‑distance niching threshold (5 % of chromosome length)
    niche_thresh = 0.05 * _ORIGIN_IDX.size

    while len(next_pop) < pop_size:
        p1 = _tournament(pop_records[:rank_depth], 3)
        p2 = _tournament(pop_records[:rank_depth], 3)
        child = uniform_crossover(p1, p2, swap_p=0.5)
        child = mutate_gauss(child, sigma, bounds)

        if all(_hamming(child, dna) > niche_thresh for dna in next_pop):
            next_pop.append(child)

    return next_pop
