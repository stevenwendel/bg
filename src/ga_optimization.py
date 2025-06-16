"""Bayesian optimisation wrapper around `ga_runner.run_ga`.

* Creates a throw‑away GA preset named ``"bayes"`` in ``GA_CONFIG`` each call.
* Uses scikit‑optimize (skopt) `gp_minimize` to search over
  – mutation rate,  
  – mutation sigma,  
  – population size (integer).
* Objective = **negative best score** (because skopt minimises).
* Results are pickled + human‑readable summary stored in a timestamped folder
  under `results/bayes_*`.

Note: This script automatically sets NUMBA_NUM_THREADS=1 to ensure reproducible results
and proper resource management.

Run:
    NUMBA_NUM_THREADS=1 python bayes_opt.py --calls 20
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
import copy
import os
import pickle
import time
from datetime import datetime

import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer

from src.constants import GA_CONFIG
from ga_runner import run_ga

# ------------------------------------------------------------------
# 1.  Evaluate a single GA configuration
# ------------------------------------------------------------------

def _run_single(cfg: dict, work_dir: Path) -> int:
    start_single = time.time()
    preset = "bayes_tmp"
    GA_CONFIG[preset] = copy.deepcopy(cfg)
    # redirect GA results into the working dir ------------------------
    os.environ["RESULTS_DIR"] = str(work_dir / "results")

    best_score = run_ga(preset, results_dir=os.environ["RESULTS_DIR"])

    # grab last line of elites.txt for the DNA ------------------------
    dna_path = Path(os.environ["RESULTS_DIR"]) / "elites.txt"
    best_dna = None
    if dna_path.exists():
        with dna_path.open() as fh:
            for ln in fh:
                pass
            best_dna = [float(x) for x in ln.strip().split(",")]

    del GA_CONFIG[preset]   # avoid clutter
    return best_score, best_dna


# ------------------------------------------------------------------
# 2.  Bayesian optimisation loop
# ------------------------------------------------------------------

def bayes_opt(n_calls: int):
    out_dir = Path("results") / f"bayes_{datetime.now():%Y%m%d_%H%M%S}"
    (out_dir / "results").mkdir(parents=True, exist_ok=True)

    max_sims = 10_000

    trial_results = []

    def objective(x):
        mut_rate, mut_sigma, pop_size = x
        pop_size = int(pop_size)
        num_gen  = max_sims // pop_size

        cfg = {
            "POP_SIZE"      : pop_size,
            "NUM_GENERATIONS": num_gen,
            "MUT_RATE"      : mut_rate,
            "MUT_SIGMA"     : mut_sigma,
            "ELITE_SIZE"    : 10,
            "DNA_BOUNDS"    : [0, 500],
            "RANK_DEPTH"    : pop_size // 2,
        }
        print("\n=== Evaluating", cfg)
        score, dna = _run_single(cfg, out_dir)
        print("→ best", score)
        print("→ dna", dna)
        trial_results.append({"cfg": cfg, "score": score, "dna": dna})
        return -score  # skopt minimises

    space = [
        Real(0.3, 0.8,    name="mut_rate"),
        Real(0.2, 2.0,    name="mut_sigma"),
        Integer(50, 400, name="pop_size"),
    ]

    result = gp_minimize(objective, space, n_calls=n_calls, random_state=None, verbose=True)
    sk_dump(result, out_dir / "skopt_result.pkl")            # safe dump
    with (out_dir / "trial_results.pkl").open("wb") as fh:   # our custom list
        pickle.dump(trial_results, fh)

    best_idx   = int((-result.fun) == max(t["score"] for t in trial_results))
    best_trial = max(trial_results, key=lambda d: d["score"])

    print("\n=== optimisation finished ===")
    print("Best params :", best_trial["params"])
    print("Best score  :", best_trial["score"])
    print("Best DNA    :", best_trial["dna"])

# ------------------------------------------------------------------
# 3.  CLI glue
# ------------------------------------------------------------------

def main():
    # Set NUMBA_NUM_THREADS=1 for reproducible results
    os.environ["NUMBA_NUM_THREADS"] = "1"
    
    ap = argparse.ArgumentParser(description="Bayesian optimisation for GA hyper‑params")
    ap.add_argument("--calls", type=int, default=10, help="number of skopt evaluations")
    args = ap.parse_args()
    t0 = time.time()
    bayes_opt(args.calls)
    print(f"\nWall‑time: {(time.time() - t0)/60:.1f} min")

if __name__ == "__main__":
    start=time.time()
    main()
    end=time.time()
    print(f"Time taken: {end-start} seconds")