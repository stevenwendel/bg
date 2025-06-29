"""
Fixed version of ga_runner.py with proper multiprocessing handling.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import List
import time
import os
import numba

from src.constants import GA_CONFIG
from src.genetic_algorithm import create_dna, decode_dna_to_matrix, spawn_next_population
from src.network import create_experiment, run_network
from src.neuron import create_neurons, prepare_neurons, _SPIKES
from src.validation import evaluate_conditions

# numba.set_num_threads(1)

def _init_worker():
    """Initialize worker process - create neurons once per process."""
    # This ensures each process has its own neuron state
    create_neurons()
    print(f"Worker process {os.getpid()} initialized with neurons")

def _score_one(dna_vec, input_waves, alpha_kernel):
    """Score one DNA sequence - simplified and correct."""
    import src.neuron as n
    
    # Each worker already has neurons created in _init_worker()
    # No need to copy or check - just use the global state in this process
    neurons = [n.Izhikevich(i, name) for i, name in enumerate(n.NEURON_NAMES)]
    W = decode_dna_to_matrix(dna_vec)
    cue_wave, go_wave = input_waves

    total = 0
    for ctl in (False, True):
        n.prepare_neurons(neurons, cue_wave, go_wave, ctl)
        n.t_pointer = 0
        run_network(neurons, W, alpha_kernel)
        key = "control" if ctl else "experimental"
        total += evaluate_conditions(n._SPIKES)[key]
    return total

def run_ga(preset: str, *, results_dir: str | None = None) -> int:
    cfg = GA_CONFIG[preset]
    pop_size    = cfg["POP_SIZE"]
    bounds      = tuple(cfg["DNA_BOUNDS"])
    generations = cfg["NUM_GENERATIONS"]

    out_dir = Path(results_dir) if results_dir else Path("results")
    out_dir.mkdir(exist_ok=True)

    _, input_waves, alpha_kernel = create_experiment()
    population: List = [create_dna(bounds) for _ in range(pop_size)]

    best_overall = 0
    
    # Use initializer to set up each worker process
    with mp.Pool(mp.cpu_count(), initializer=_init_worker) as pool:
        for gen in range(generations):
            fitness = pool.map(partial(_score_one, input_waves=input_waves, alpha_kernel=alpha_kernel), population)
            best_score = max(fitness)
            best_overall = max(best_overall, best_score)
            best_idx = fitness.index(best_score)
            best_dna = population[best_idx]
            print(f"Gen {gen:03d} | best {best_score:4d} | avg {sum(fitness)/pop_size:.1f} \n>>>best DNA: {best_dna.tolist()}")

            # write elites
            elites = [dna for dna, f in zip(population, fitness) if f >= 735]
            if elites:
                with open(out_dir / "elites.txt", "a") as fh:
                    for dna in elites:
                        fh.write(",".join(map(str, dna.tolist())) + "\n")

            pop_records = [{"dna": d, "dna_score": s} for d, s in zip(population, fitness)]
            population  = spawn_next_population(pop_records, cfg, gen)
    return best_overall

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
        run_ga(args.preset)

        end = time.time()
        print((end-start))

    print(f"Total time: {time.time() - start0}") 