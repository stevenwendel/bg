"""
Guide for Capturing DNA Data in Bayesian Optimization
====================================================

This file shows how to modify your code to capture the best DNA from each
genetic algorithm run, so you can analyze not just the parameters but also
the actual neural network weights that achieved the best scores.
"""

import pickle
import numpy as np
from pathlib import Path

# ============================================================================
# PROBLEM: Current Implementation
# ============================================================================

"""
Currently, your Bayesian optimization only captures:
- GA parameters (POP_SIZE, MUT_RATE, etc.)
- Final score
- DNA: None (missing!)

This means you can't analyze what the actual best neural network weights were.
"""

# ============================================================================
# SOLUTION: Modified GA Runner
# ============================================================================

def run_ga_with_dna_capture(preset: str, *, results_dir: str | None = None) -> tuple[int, np.ndarray]:
    """
    Modified version of run_ga that returns both score and best DNA
    
    Returns:
        tuple: (best_score, best_dna)
    """
    from src.constants import GA_CONFIG
    from src.genetic_algorithm import create_dna, decode_dna_to_matrix, spawn_next_population
    from src.network import create_experiment, run_network
    from src.neuron import create_neurons, prepare_neurons, _SPIKES
    from src.validation import evaluate_conditions
    import multiprocessing as mp
    from copy import deepcopy
    from functools import partial
    from pathlib import Path
    from typing import List
    import time
    import os
    import numba
    
    cfg = GA_CONFIG[preset]
    pop_size    = cfg["POP_SIZE"]
    bounds      = tuple(cfg["DNA_BOUNDS"])
    generations = cfg["NUM_GENERATIONS"]

    out_dir = Path(results_dir) if results_dir else Path("results")
    out_dir.mkdir(exist_ok=True)

    _, input_waves, alpha_kernel = create_experiment()
    population: List = [create_dna(bounds) for _ in range(pop_size)]

    best_overall = 0
    best_dna_overall = None  # NEW: Track the best DNA
    
    with mp.Pool(mp.cpu_count(), initializer=_init_worker) as pool:
        for gen in range(generations):
            fitness = pool.map(partial(_score_one, input_waves=input_waves, alpha_kernel=alpha_kernel), population)
            best_score = max(fitness)
            best_idx = fitness.index(best_score)
            best_dna = population[best_idx]
            
            # NEW: Update best overall DNA if we have a new best score
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
    
    # NEW: Return both score and DNA
    return best_overall, best_dna_overall

# ============================================================================
# SOLUTION: Modified Bayesian Optimization
# ============================================================================

def objective_function_with_dna(params):
    """
    Modified objective function that captures DNA
    
    This would replace your current objective function in the Bayesian optimization
    """
    # Extract parameters
    pop_size = int(params[0])
    num_generations = int(params[1])
    mut_rate = params[2]
    mut_sigma = params[3]
    elite_size = int(params[4])
    rank_depth = int(params[5])
    
    # Create temporary config
    temp_config = {
        "POP_SIZE": pop_size,
        "NUM_GENERATIONS": num_generations,
        "MUT_RATE": mut_rate,
        "MUT_SIGMA": mut_sigma,
        "ELITE_SIZE": elite_size,
        "RANK_DEPTH": rank_depth,
        "DNA_BOUNDS": [0, 500]
    }
    
    # Run GA with DNA capture
    best_score, best_dna = run_ga_with_dna_capture(temp_config)
    
    # Return both score and DNA
    return best_score, best_dna

# ============================================================================
# SOLUTION: Modified Trial Results Structure
# ============================================================================

def create_trial_result_with_dna(cfg, score, dna):
    """
    Create a trial result that includes DNA
    """
    return {
        'cfg': cfg,
        'score': score,
        'dna': dna.tolist() if dna is not None else None  # Convert numpy array to list for JSON serialization
    }

# Example usage:
"""
# In your Bayesian optimization loop:
for i in range(n_trials):
    # Get next parameters from optimizer
    next_params = optimizer.ask()
    
    # Run GA and get both score and DNA
    score, dna = objective_function_with_dna(next_params)
    
    # Tell optimizer about the result
    optimizer.tell(next_params, score)
    
    # Store trial result with DNA
    trial_result = create_trial_result_with_dna(cfg, score, dna)
    trial_results.append(trial_result)
    
    # Save to file
    with open('trial_results_with_dna.pkl', 'wb') as f:
        pickle.dump(trial_results, f)
"""

# ============================================================================
# ANALYSIS WITH DNA DATA
# ============================================================================

def analyze_trials_with_dna(filepath):
    """
    Analyze trials that include DNA data
    """
    with open(filepath, 'rb') as f:
        trial_results = pickle.load(f)
    
    print("=== TRIAL ANALYSIS WITH DNA ===")
    
    for i, trial in enumerate(trial_results):
        print(f"\nTrial {i+1}:")
        print(f"  Score: {trial['score']}")
        print(f"  Parameters: {trial['cfg']}")
        
        if trial['dna'] is not None:
            dna_array = np.array(trial['dna'])
            print(f"  DNA length: {len(dna_array)}")
            print(f"  DNA mean: {dna_array.mean():.2f}")
            print(f"  DNA std: {dna_array.std():.2f}")
            print(f"  DNA range: [{dna_array.min():.2f}, {dna_array.max():.2f}]")
            
            # You could also decode the DNA to see the actual weights
            # from src.genetic_algorithm import decode_dna_to_matrix
            # weight_matrix = decode_dna_to_matrix(dna_array)
            # print(f"  Weight matrix shape: {weight_matrix.shape}")
        else:
            print("  DNA: None (not captured)")
    
    # Find best DNA
    best_trial = max(trial_results, key=lambda x: x['score'])
    print(f"\n🏆 BEST TRIAL (Score: {best_trial['score']}):")
    if best_trial['dna'] is not None:
        best_dna = np.array(best_trial['dna'])
        print(f"  Best DNA: {best_dna.tolist()}")
        print(f"  You can use this DNA to recreate the best neural network!")

# ============================================================================
# QUICK FIX FOR EXISTING CODE
# ============================================================================

def quick_fix_for_existing_runs():
    """
    If you want to capture DNA in your existing Bayesian optimization setup,
    here's the minimal change needed:
    """
    
    # 1. Modify your objective function to return DNA
    def objective_function(params):
        # ... your existing parameter extraction ...
        
        # Instead of just returning the score:
        # return run_ga(config)
        
        # Return both score and DNA:
        score, dna = run_ga_with_dna_capture(config)
        return score, dna
    
    # 2. Modify your optimization loop:
    """
    for i in range(n_trials):
        next_params = optimizer.ask()
        score, dna = objective_function(next_params)  # Now returns both
        
        optimizer.tell(next_params, score)
        
        # Store with DNA
        trial_result = {
            'cfg': config,
            'score': score,
            'dna': dna.tolist() if dna is not None else None
        }
        trial_results.append(trial_result)
    """

# ============================================================================
# EDUCATIONAL: Why DNA Analysis is Important
# ============================================================================

def why_dna_analysis_matters():
    """
    Explain why capturing DNA is valuable
    """
    print("""
WHY DNA ANALYSIS MATTERS
=======================

1. **Reproducibility**: You can recreate the exact neural network that achieved the best score

2. **Transfer Learning**: The best DNA from one optimization run can be used as a starting point for other runs

3. **Pattern Analysis**: You can analyze what types of weight configurations work best:
   - Are certain connections always strong/weak?
   - Are there patterns in the weight distributions?
   - Do successful networks have similar architectures?

4. **Ensemble Methods**: You can combine multiple good DNA solutions to create ensemble networks

5. **Debugging**: If something goes wrong, you can examine the exact weights that caused issues

6. **Research Insights**: Understanding what weight configurations work can inform your understanding of the neural dynamics

EXAMPLE ANALYSIS QUESTIONS:
- What is the distribution of weights in the best-performing networks?
- Are there specific connections that are consistently strong across good solutions?
- Do the best solutions have similar weight patterns?
- Can we identify "signature" weight configurations that predict good performance?
""")

if __name__ == "__main__":
    print("DNA Capture Guide")
    print("=" * 50)
    why_dna_analysis_matters()
    
    print("\n" + "=" * 50)
    print("IMPLEMENTATION STEPS:")
    print("1. Modify run_ga() to return both score and best DNA")
    print("2. Update your objective function to capture DNA")
    print("3. Modify your trial result structure to include DNA")
    print("4. Update your analysis scripts to examine DNA patterns")
    print("\nSee the code examples above for specific implementations.") 