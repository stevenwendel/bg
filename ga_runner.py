"""Genetic‑algorithm driver: evolve BG‑network synaptic weights.

Run with:
    python ga_runner.py --preset small

Key points
----------
* **Pool initializer (`_init_worker`)** builds a neuron template once *inside*
  each subprocess; this guarantees that `src.neuron._INPUT` is allocated in
  the child, preventing the `NoneType` crashes you saw.
* Each `_score_one` simply `deepcopy`s that template – cheap and safe.
* Pool size defaults to all logical CPU cores (`mp.cpu_count()`).
* DNA scoring ≥ 735 are appended to `results/elites.txt`.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import List
import time
import numba

# Configure Numba to use only 1 thread
numba.set_num_threads(1)

from src.constants import GA_CONFIG
from src.genetic_algorithm import create_dna, decode_dna_to_matrix, spawn_next_population
from src.network import create_experiment, run_network
from src.neuron import create_neurons, prepare_neurons, _SPIKES
from src.validation import evaluate_conditions

# ------------------------------------------------------------------
# 1.  Per‑process neuron template
# ------------------------------------------------------------------
_template = None  # set in each worker by _init_worker


def _init_worker():
    """Run once in every subprocess: allocate _INPUT and compile Numba."""
    global _template
    _template = create_neurons()


# ------------------------------------------------------------------
# 2.  Chromosome scorer
# ------------------------------------------------------------------

def _score_one(dna_vec, input_waves, alpha_kernel):
    import src.neuron as n  # local module inside worker

    # Re‑initialise template if spawn skipped initializer (edge case)
    global _template
    if _template is None or n._INPUT is None:
        _template = create_neurons()

    neurons = deepcopy(_template)  # fast copy (arrays only)
    W       = decode_dna_to_matrix(dna_vec)
    cue_wave, go_wave = input_waves

    total = 0
    for ctl in (False, True):
        n.prepare_neurons(neurons, cue_wave, go_wave, ctl)
        n.t_pointer = 0
        run_network(neurons, W, alpha_kernel)
        key = "control" if ctl else "experimental"
        total += evaluate_conditions(n._SPIKES)[key]
    return total


# ------------------------------------------------------------------
# 3.  GA driver
# ------------------------------------------------------------------

def run_ga(preset: str):
    cfg = GA_CONFIG[preset]
    pop_size    = cfg["POP_SIZE"]
    bounds      = tuple(cfg["DNA_BOUNDS"])
    generations = cfg["NUM_GENERATIONS"]

    _, input_waves, alpha_kernel = create_experiment()
    population: List = [create_dna(bounds) for _ in range(pop_size)]

    with mp.Pool(mp.cpu_count(), initializer=_init_worker) as pool:
        for gen in range(generations):
            fitness = pool.map(partial(_score_one, input_waves=input_waves, alpha_kernel=alpha_kernel), population)
            best_score = max(fitness)
            best_idx = fitness.index(best_score)
            best_dna = population[best_idx]
            print(f"Gen {gen:03d} | best {best_score:4d} | avg {sum(fitness)/pop_size:.1f}")
            print(f"Best DNA: {best_dna.tolist()}")

            # save elites
            elites = [dna for dna, f in zip(population, fitness) if f >= 735]
            if elites:
                Path("results").mkdir(exist_ok=True)
                with open("results/elites.txt", "a") as fh:
                    for dna in elites:
                        fh.write(",".join(map(str, dna.tolist())) + "\n")

            pop_records = [{"dna": d, "dna_score": s} for d, s in zip(population, fitness)]
            population  = spawn_next_population(pop_records, cfg, gen)


# ------------------------------------------------------------------
# 4.  CLI entry‑point
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Run GA for BG network")
    ap.add_argument("--preset", choices=GA_CONFIG.keys(), default="small")
    args = ap.parse_args()
    run_ga(args.preset)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print((end-start))