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
from typing import Tuple, List

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
from skopt.utils import dump as sk_dump

from src.constants import GA_CONFIG
from ga_runner import run_ga

# ------------------------------------------------------------------
# 1.  Evaluate a single GA configuration
# ------------------------------------------------------------------

def _run_single(cfg: dict, work_dir: Path) -> Tuple[int, List[float]]:
    start_single = time.time()
    preset = "bayes_tmp"
    GA_CONFIG[preset] = copy.deepcopy(cfg)
    # redirect GA results into the working dir ------------------------
    os.environ["RESULTS_DIR"] = str(work_dir / "results")

    # Now run_ga returns both best_score and best_dna
    best_score, best_dna = run_ga(preset, results_dir=os.environ["RESULTS_DIR"])

    del GA_CONFIG[preset]   # avoid clutter
    return best_score, best_dna


# ------------------------------------------------------------------
# 2.  Bayesian optimisation loop
# ------------------------------------------------------------------

def create_objective(max_sims: int, out_dir: Path, trial_results: list):
    """Create an objective function for Bayesian optimization."""
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
        
        # Save current trial info to text file
        with (out_dir / "trial_summary.txt").open("a") as fh:
            fh.write(f"\n{'='*60}\n")
            fh.write(f"Trial {len(trial_results)}\n")
            fh.write(f"Timestamp: {datetime.now()}\n")
            fh.write(f"Parameters:\n")
            for key, value in cfg.items():
                fh.write(f"  {key}: {value}\n")
            fh.write(f"Best Score: {score}\n")
            fh.write(f"Best DNA: {dna}\n")
            fh.write(f"{'='*60}\n")
        
        return -score  # skopt minimises
    
    return objective

def bayes_opt(n_calls: int):
    out_dir = Path("results") / f"bayes_{datetime.now():%Y%m%d_%H%M%S}"
    (out_dir / "results").mkdir(parents=True, exist_ok=True)

    max_sims = 100_000
    trial_results = []

    # Initialize the summary text file
    with (out_dir / "trial_summary.txt").open("w") as fh:
        fh.write(f"BAYESIAN OPTIMIZATION SUMMARY\n")
        fh.write(f"{'='*80}\n")
        fh.write(f"Start Time: {datetime.now()}\n")
        fh.write(f"Number of Calls: {n_calls}\n")
        fh.write(f"Max Simulations: {max_sims}\n")
        fh.write(f"Output Directory: {out_dir}\n")
        fh.write(f"{'='*80}\n\n")

    # Create the objective function
    objective = create_objective(max_sims, out_dir, trial_results)

    space = [
        Real(0.2, 0.8,    name="mut_rate"),
        Real(0.2, 20.0,    name="mut_sigma"),
        Integer(500, 1000, name="pop_size"),
    ]

    result = gp_minimize(objective, space, n_calls=n_calls, random_state=None, verbose=True)
    
    # Remove the objective function from the result to avoid pickling issues
    if hasattr(result, 'specs'):
        result.specs['args']['func'] = None
    
    sk_dump(result, out_dir / "skopt_result.pkl")            # safe dump
    with (out_dir / "trial_results.pkl").open("wb") as fh:   # our custom list
        pickle.dump(trial_results, fh)

    best_idx   = int((-result.fun) == max(t["score"] for t in trial_results))
    best_trial = max(trial_results, key=lambda d: d["score"])

    print("\n=== optimisation finished ===")
    print("Best params :", best_trial["cfg"])
    print("Best score  :", best_trial["score"])
    print("Best DNA    :", best_trial["dna"])
    
    # Write final summary to text file
    with (out_dir / "trial_summary.txt").open("a") as fh:
        fh.write(f"\n{'='*80}\n")
        fh.write(f"OPTIMIZATION COMPLETE\n")
        fh.write(f"Total Trials: {len(trial_results)}\n")
        fh.write(f"Best Trial: {best_idx + 1}\n")
        fh.write(f"Best Score: {best_trial['score']}\n")
        fh.write(f"Best Parameters:\n")
        for key, value in best_trial["cfg"].items():
            fh.write(f"  {key}: {value}\n")
        fh.write(f"Best DNA: {best_trial['dna']}\n")
        fh.write(f"{'='*80}\n")
        
        # Write sorted results by score (best to worst)
        fh.write(f"\nALL TRIALS RANKED BY SCORE (BEST TO WORST):\n")
        fh.write(f"{'='*80}\n")
        sorted_trials = sorted(trial_results, key=lambda d: d["score"], reverse=True)
        for i, trial in enumerate(sorted_trials):
            fh.write(f"\nRank {i+1} (Score: {trial['score']}):\n")
            fh.write(f"  Parameters: {trial['cfg']}\n")
            fh.write(f"  DNA: {trial['dna']}\n")
            fh.write(f"  {'-'*40}\n")

# ------------------------------------------------------------------
# 3.  CLI glue
# ------------------------------------------------------------------

def main():
    # Set NUMBA_NUM_THREADS=1 for reproducible results
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["NUMBA_DISABLE_JIT"] = "1"

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