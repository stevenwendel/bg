"""Genetic‑algorithm driver (updated): now returns best score and lets caller
specify a custom **results_dir**.

Signature
---------
    best = run_ga(preset:str, results_dir: str|None = None)

* If *results_dir* is given, elites are written there; otherwise to ./results.
* Returns the best TOTAL score across all generations so Bayesian optimiser
  can consume it directly.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import List, Tuple
import time
import os
import numba

from src.constants import GA_CONFIG
from src.genetic_algorithm import create_dna, decode_dna_to_matrix, spawn_next_population
from src.network import create_experiment, run_network
from src.neuron import create_neurons, prepare_neurons, _SPIKES
from src.validation import evaluate_conditions

# numba.set_num_threads(1)

_template = None

def _init_worker():
    global _template
    _template = create_neurons()


def _score_one(dna_vec, input_waves, alpha_kernel):
    import src.neuron as n
    global _template
    if _template is None or n._INPUT is None:
        _template = create_neurons()

    neurons = deepcopy(_template)
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


def run_ga(preset: str, *, results_dir: str | None = None) -> Tuple[int, List[float]]:
    """
    Run genetic algorithm and return best score and best DNA.
    
    Returns:
        Tuple[int, List[float]]: (best_score, best_dna)
    """
    cfg = GA_CONFIG[preset]
    pop_size    = cfg["POP_SIZE"]
    bounds      = tuple(cfg["DNA_BOUNDS"])
    generations = cfg["NUM_GENERATIONS"]

    out_dir = Path(results_dir) if results_dir else Path("results")
    out_dir.mkdir(exist_ok=True)

    _, input_waves, alpha_kernel = create_experiment()
    population: List = [create_dna(bounds) for _ in range(pop_size)]

    best_overall = 0
    best_dna_overall = None  # Track the best DNA across all generations
    
    with mp.Pool(mp.cpu_count(), initializer=_init_worker) as pool:
        for gen in range(generations):
            fitness = pool.map(partial(_score_one, input_waves=input_waves, alpha_kernel=alpha_kernel), population)
            best_score = max(fitness)
            best_idx = fitness.index(best_score)
            best_dna = population[best_idx]
            
            # Update best overall if we have a new best score
            if best_score > best_overall:
                best_overall = best_score
                best_dna_overall = best_dna.copy()  # Make a copy to preserve it
            
            print(f"Gen {gen:03d} | best {best_score:4d} | avg {sum(fitness)/pop_size:.1f} \n>>>best DNA: {best_dna.tolist()}")

            # write elites
            elites = [dna for dna, f in zip(population, fitness) if f >= 735]
            if elites:
                with open(out_dir / "elites.txt", "a") as fh:
                    for dna in elites:
                        fh.write(",".join(map(str, dna.tolist())) + "\n")

            pop_records = [{"dna": d, "dna_score": s} for d, s in zip(population, fitness)]
            population  = spawn_next_population(pop_records, cfg, gen)
    
    # Return both the best score and the best DNA
    return best_overall, best_dna_overall.tolist()


if __name__ == "__main__":
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    start0 = time.time()
    for i in range(5):
        print(f"Run {i}")
        start = time.time()
        parser = argparse.ArgumentParser()
        parser.add_argument("--preset", choices=GA_CONFIG.keys(), default="large")
        args = parser.parse_args()
        best_score, best_dna = run_ga(args.preset)
        print(f"Best score: {best_score}")
        print(f"Best DNA: {best_dna}")

        end = time.time()
        print((end-start))

    print(f"Total time: {time.time() - start0}")