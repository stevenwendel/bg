"""Entry‑point for batch evaluation of DNA strings / weight matrices.

This script wires together *all* modernised components:

📦  src.utils.create_alpha_array
🧠  src.neuron (vectorised)
🔗  src.network (fast runner)
🔬  src.validation (criteria + scoring)

It supports three common workflows:
  1️⃣  Evaluate **one** DNA string ("--dna …")
  2️⃣  Evaluate **one** .npy weight matrix ("--matrix …")
  3️⃣  Evaluate **many** DNA strings from a text‑file ("--dna-list file.txt")

Each run prints the experimental & control scores plus the total, and writes a
CSV summary if more than one individual is evaluated.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np

from src.utils import load_dna, create_alpha_array  # type: ignore – load_dna assumed present
from src.neuron import create_neurons, prepare_neurons
from src.network import create_experiment, run_network
from src.validation import define_criteria, evaluate_dna
from src.constants import CRITERIA_NAMES, BIN_SIZE, TMAX

# --------------------------------------------------------------------
# Helper to run & score a single individual
# --------------------------------------------------------------------

def _simulate_and_score(weight_matrix: np.ndarray, cue_wave, go_wave, alpha_kernel, criteria):
    neurons = create_neurons()
    prepare_neurons(neurons, cue_wave, go_wave, control=False)  # will reset inside evaluate_dna
    scores, _ = evaluate_dna(
        dna_matrix=weight_matrix,
        neurons=neurons,
        alpha_array=alpha_kernel,
        input_waves=[cue_wave, go_wave],
        criteria=criteria,
    )
    total = sum(scores.values())
    return total, scores

# --------------------------------------------------------------------
# Main CLI
# --------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Batch evaluator for BG‑network DNA strings/matrices.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dna", help="Evaluate a single DNA string (e.g. ACTG…)")
    g.add_argument("--matrix", type=Path, help="Evaluate a single .npy weight matrix")
    g.add_argument("--dna-list", type=Path, help="Text file with one DNA string per line")
    p.add_argument("--out", type=Path, default=None, help="Optional CSV output path for batch runs")
    args = p.parse_args()

    # Build experiment constants once --------------------------------
    _, input_waves, alpha_kernel = create_experiment()
    cue_wave, go_wave = input_waves
    criteria = define_criteria()

    results = []  # list of (id, total, exp, ctl)

    def _add_result(identifier: str, matrix: np.ndarray):
        total, scores = _simulate_and_score(matrix, cue_wave, go_wave, alpha_kernel, criteria)
        print(f"{identifier:>20}:  TOTAL={total:4d}  experimental={scores['experimental']:4d}  control={scores['control']:4d}")
        results.append((identifier, total, scores['experimental'], scores['control']))

    # ----------------------------------------------------------------
    # Dispatch based on CLI mode
    # ----------------------------------------------------------------
    if args.dna:
        _add_result("dna", load_dna(args.dna))
    elif args.matrix:
        _add_result(args.matrix.stem, np.load(args.matrix))
    else:  # batch list
        with open(args.dna_list) as f:
            for idx, line in enumerate(f, 1):
                dna = line.strip()
                if not dna:
                    continue
                _add_result(f"dna_{idx}", load_dna(dna))

    # ----------------------------------------------------------------
    # Optional CSV export
    # ----------------------------------------------------------------
    if args.out and results:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["id", "total", "experimental", "control"])
            writer.writerows(results)
        print(f"\nSaved batch summary to {args.out}")

if __name__ == "__main__":
    main()
