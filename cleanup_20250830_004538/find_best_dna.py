#!/usr/bin/env python3

import pickle
import numpy as np

# Load the aggregated results
results_path = "/Users/stevenwendel/Documents/GitHub/bg/results/multiple_runs_H_opt4_20250815_180350/run_001/aggregated_results.pkl"

with open(results_path, 'rb') as f:
    results = pickle.load(f)

print("Searching for highest scoring DNA...")

# Extract all tested DNA with their scores
all_dna_tested = results['all_dna_tested']
print(f"Found {len(all_dna_tested)} tested individuals")

# Find the highest scoring individual
highest_fitness = -np.inf
best_dna = None
best_individual = None

for individual in all_dna_tested:
    score = individual['total_score']
    if score > highest_fitness:
        highest_fitness = score
        best_dna = individual['dna']
        best_individual = individual

print(f"\nHighest fitness found: {highest_fitness}")
print(f"Experimental score: {best_individual['exp_score']}")
print(f"Control score: {best_individual['cont_score']}")
print(f"Process ID: {best_individual['process_id']}")
print(f"Generation: {best_individual['generation']}")
print(f"Individual ID: {best_individual['individual_id']}")
print(f"DNA shape: {np.array(best_dna).shape}")

# Save the best DNA and individual info to a file
with open('/Users/stevenwendel/Documents/GitHub/bg/best_dna.pkl', 'wb') as f:
    pickle.dump(best_individual, f)

print("Saved best individual to best_dna.pkl")
print(f"Best DNA: {best_dna}")