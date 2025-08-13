#!/usr/bin/env python3
"""
Multiprocessing version of v2.ipynb genetic algorithm.

This script runs separate genetic algorithm populations on each CPU core,
allowing for parallel evolution of different populations. Each process
maintains its own population and saves all DNA scores and vectors tested.

Key features:
- One GA population per CPU core for true parallelism
- All DNA vectors and scores saved to pickle files
- Configurable GA parameters
- Progress reporting and timing
- Results aggregation across all processes
"""

import os
import time
import pickle
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import random
import numpy as np

# Set environment variables before importing numba
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"  
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_DISABLE_JIT"] = "0"

from numba import njit, prange, float32, uint8
from src.constants import *

# Global simulation parameters (computed once)
N = len(NEURON_NAMES)
ALPHA_L = 250
td = np.arange(1, ALPHA_L + 1, dtype=np.float32)
alpha = (td / 30) * np.exp((30 - td) / 30)   

cue_wave = np.zeros(TMAX, dtype=np.float32)
go_wave = np.zeros_like(cue_wave)
cue_wave[EPOCHS['sample'][0]:EPOCHS['sample'][1]] = CUE_STRENGTH
go_wave[EPOCHS['response'][0]:EPOCHS['response'][0] + GO_DURATION] = GO_STRENGTH

pass_ids = np.array([NEURON_NAMES.index(x) for x in ["VMresp", "ALMresp", "SNR3"]])

# Create criterion matrices
conditions = []
for condition in CRITERIA:
    condition_criteria = []
    for neuron_name, neuron in CRITERIA[condition].items():
        idx = NEURON_NAMES.index(neuron_name)
        baseline = np.ones(TMAX, np.uint8) if neuron_name in TONICALLY_ACTIVE_NEURONS else np.zeros(TMAX, np.uint8)
        start = neuron["interval"][0]
        end = neuron["interval"][1]
        target_status = neuron["io"]
        
        if target_status == "off":
            baseline[start:end] = 0
        elif target_status == "on":
            baseline[start:end] = 1

        baseline = baseline.reshape(TMAX//BIN_SIZE, BIN_SIZE)
        baseline = np.sum(baseline, axis=1, dtype=np.uint32)
        baseline = (baseline != 0).astype(np.uint8)
        condition_criteria.append((neuron_name, idx, baseline))
    
    condition_criteria = sorted(condition_criteria, key=lambda tup: tup[1])
    conditions.append(condition_criteria)

crit_Exp, crit_Cont = conditions
crit_indices = np.array([neu[1] for neu in crit_Cont])
crit_Exp = np.vstack([neu[2] for neu in crit_Exp])
crit_Cont = np.vstack([neu[2] for neu in crit_Cont])

# Genetic algorithm operators
_ORIGIN_IDX = np.array([NEURON_NAMES.index(o) for o, _ in ACTIVE_SYNAPSES], dtype=np.int16)
_TARGET_IDX = np.array([NEURON_NAMES.index(t) for _, t in ACTIVE_SYNAPSES], dtype=np.int16)
_INHIB_MASK = np.isin(_ORIGIN_IDX, [NEURON_NAMES.index(n) for n in INHIBITORY_NEURONS])


# ============================================================================
# Core simulation functions (from v2.ipynb)
# ============================================================================

@njit(parallel=False, fastmath=True, cache=True)
def step_kernel(V, U, Ibuf, t_ptr, a, b, vreset, d, k, vr, vt, vpeak, C, E, W, alpha):
    """Neural network step kernel with Izhikevich dynamics."""
    n, L = V.size, alpha.size
    spk = np.zeros(n, dtype=np.uint8)

    # integrate
    for i in range(n):
        I = Ibuf[t_ptr, i]
        dV = (k[i]*(V[i]-vr[i])*(V[i]-vt[i]) - U[i] + I + E[i]) / C[i]
        dU = a[i]*(b[i]*(V[i]-vr[i]) - U[i])
        V[i] += dV
        U[i] += dU
        if V[i] >= vpeak[i]:
            V[i] = vreset[i]
            U[i] += d[i]
            spk[i] = 1

    # distribute PSC
    if np.sum(spk) > 0:
        post_I = spk.astype(np.float32) @ W
        t_next = (t_ptr + 1) % L
        for k_shift in range(L):
            Ibuf[(t_next + k_shift) % L, :] += post_I * alpha[k_shift]

    Ibuf[t_ptr,:] = 0.0
    return spk, (t_ptr + 1) % L


@njit(fastmath=True, cache=True)
def score_bin(curr_bin_results, crit_matrix, crit_indices, bin_idx, pass_ids):
    """Score a single time bin against criteria."""
    score = 0
    for i in range(len(crit_indices)):
        idx = crit_indices[i]
        if curr_bin_results[idx] == crit_matrix[i, bin_idx]:
            score += 1
        elif (bin_idx * BIN_SIZE > 3500) and (idx in pass_ids):
            score += 1
    return score


@njit(fastmath=True, cache=True)
def simulate(W, a, b, vreset, d, k, vr, vt, vpeak, C, E, alpha, cue_wave, go_wave, 
            crit_Exp, crit_Cont, crit_indices, pass_ids, tmax, control, return_full):
    """Simulate neural network and return fitness score."""
    W = np.ascontiguousarray(W)
    V = np.full(N, -60.0, np.float32)
    U = np.zeros_like(V, np.float32)
    Ibuf = np.zeros((ALPHA_L, N), dtype=np.float32)
    HIST = np.zeros((N, BIN_SIZE), np.uint8)
    temp_full_hist = None
    
    if return_full:
        temp_full_hist = np.zeros((N, tmax), np.uint8)

    score = 0
    t_ptr = 0
    bin = 0

    for t in range(tmax):
        if control == False:
            Ibuf[t_ptr,0] += cue_wave[t]
        Ibuf[t_ptr,7] += go_wave[t]
    
        spk, t_ptr = step_kernel(V, U, Ibuf, t_ptr, a, b, vreset, d, k, vr, vt, vpeak, C, E, W, alpha)

        if return_full and temp_full_hist is not None:
            temp_full_hist[:,t] = spk 

        cidx = t % BIN_SIZE
        HIST[:,cidx] = spk
        
        if cidx == (BIN_SIZE - 1):
            curr_bin_results = (np.sum(HIST, axis=1) >= 1).astype(np.uint8)
            crits = crit_Exp if (control == False) else crit_Cont
            score += score_bin(curr_bin_results, crits, crit_indices, bin, pass_ids)
            bin += 1

    return score, (temp_full_hist if return_full else None)


@njit
def create_matrices(dna_vectors: np.array, conn_map: np.array, num_neurons: int):
    """Convert DNA vectors to weight matrices."""   
    N = num_neurons
    Ws = np.zeros((len(dna_vectors), N, N), dtype=np.float32)
    for idx, vector in enumerate(dna_vectors):
        for w, conn in zip(vector, conn_map):
            Ws[idx, conn[0], conn[1]] += w
    return Ws


@njit(parallel=False, fastmath=True, cache=True)
def evaluate_population(population_vectors, conn_map, N, a, b, vreset, d, k, vr, vt, vpeak, C, E, 
                       alpha, cue_wave, go_wave, crit_Exp, crit_Cont, crit_indices, pass_ids, tmax, return_full):
    """Evaluate fitness of entire population."""
    population_matrices = create_matrices(population_vectors, conn_map, N)
    
    vectors_scores = np.zeros((len(population_matrices), 3), dtype=np.int32)
    for idx in range(len(population_vectors)):
        exp_score, _ = simulate(population_matrices[idx], a, b, vreset, d, k, vr, vt, vpeak, C, E,
                               alpha, cue_wave, go_wave, crit_Exp, crit_Cont, crit_indices, pass_ids,
                               tmax, False, return_full)
        cont_score, _ = simulate(population_matrices[idx], a, b, vreset, d, k, vr, vt, vpeak, C, E,
                                alpha, cue_wave, go_wave, crit_Exp, crit_Cont, crit_indices, pass_ids,
                                tmax, True, return_full)
        vectors_scores[idx,0] = idx
        vectors_scores[idx,1] = exp_score
        vectors_scores[idx,2] = cont_score    

    return vectors_scores


# ============================================================================
# Population and genetic algorithm functions
# ============================================================================

def initialize_population(size, upper_bound, synapses, inhibited):
    """Initialize random population of DNA vectors."""
    base_vector = np.ones(len(synapses), dtype=np.int32)
    for idx, conn in enumerate(synapses):
        if conn[0] in inhibited:
            base_vector[idx] *= -1
    
    vectors = []
    while len(vectors) < size:
        random_values = np.random.randint(0, upper_bound, len(base_vector), dtype=np.int32)
        random_vector = base_vector * random_values
        random_vector = random_vector.astype(np.int32)
        vectors.append(random_vector)      
    
    return np.array(vectors, dtype=np.int32)


def initialize_connection_mapping(synapses: list, neuron_names: list) -> np.ndarray:
    """Create connection mapping from synapses to neuron indices."""
    indices = np.zeros((len(synapses), 2), dtype=np.int32)
    for idx, conn in enumerate(synapses):
        indices[idx, 0] = neuron_names.index(conn[0])  # pre-synaptic
        indices[idx, 1] = neuron_names.index(conn[1])  # post-synaptic
    return indices


def uniform_crossover(p1: np.ndarray, p2: np.ndarray, swap_p: float = 0.5) -> np.ndarray:
    """Per-gene uniform crossover."""
    mask = np.random.rand(p1.size) < swap_p
    child = np.where(mask, p1, p2)
    return child.astype(np.int32)


def mutate_gauss(dna: np.ndarray, sigma: float, bounds: Tuple[int, int]) -> np.ndarray:
    """Gaussian mutation with automatic rounding and sign fix."""
    dna_mut = dna.astype(np.float32) + np.random.normal(0, sigma, size=dna.size) * dna.astype(np.float32)
    low, high = bounds
    dna_mut = np.clip(np.round(dna_mut), -high, high)
    dna_mut[_INHIB_MASK] = -np.abs(dna_mut[_INHIB_MASK])
    dna_mut[~_INHIB_MASK] = np.abs(dna_mut[~_INHIB_MASK])
    return dna_mut.astype(np.int32)


def _tournament(pop: List[dict], k: int) -> np.ndarray:
    """Return winner DNA from k-sized tournament (higher score wins)."""
    contenders = random.sample(pop, k)
    return max(contenders, key=lambda r: r["dna_score"])["dna"]


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    """Calculate Hamming distance between two DNA vectors."""
    return int(np.sum(a != b))


def spawn_next_population(pop_records: List[dict], cfg: dict) -> List[np.ndarray]:
    """Generate next population using tournament selection, crossover, and mutation."""
    pop_size = cfg["POP_SIZE"]
    bounds = tuple(cfg["DNA_BOUNDS"])
    elite_n = cfg["ELITE_SIZE"]
    rank_depth = cfg["RANK_DEPTH"]
    sigma = cfg["MUT_SIGMA"]
    mut_rate = cfg["MUT_RATE"]

    pop_records.sort(key=lambda r: r["dna_score"], reverse=True)
    elites = [r["dna"] for r in pop_records[:elite_n]]
    next_pop = elites.copy()

    # Niching threshold (1% of chromosome length)
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


# ============================================================================
# Worker process function
# ============================================================================

def worker_process(process_id: int, config_name: str, num_generations: int, seed_offset: int, 
                  results_dir: Path, shared_counter, lock) -> Dict:
    """
    Main worker function that runs a genetic algorithm in a separate process.
    
    Args:
        process_id: Unique ID for this process
        config_name: GA configuration name from GA_CONFIG
        num_generations: Number of generations to run
        seed_offset: Random seed offset for this process
        results_dir: Directory to save results
        shared_counter: Shared counter for progress reporting
        lock: Lock for thread-safe counter updates
    
    Returns:
        Dictionary with process results
    """
    
    # Set unique random seed for this process
    np.random.seed(int(time.time() * 1000) % (2**32) + seed_offset + process_id)
    random.seed(np.random.randint(0, 2**32))
    
    print(f"Process {process_id}: Starting GA with config '{config_name}'")
    
    try:
        # Get configuration
        cfg = GA_CONFIG[config_name]
        
        # Initialize population and connection mapping
        pop = initialize_population(cfg["POP_SIZE"], cfg["DNA_BOUNDS"][1], 
                                   ACTIVE_SYNAPSES, INHIBITORY_NEURONS)
        conn_map = initialize_connection_mapping(ACTIVE_SYNAPSES, NEURON_NAMES)
        
        # Storage for all DNA tested and their scores
        all_tested_dna = []
        all_scores = []
        generation_stats = []
        
        # Evolution loop
        for gen in range(num_generations):
            gen_start = time.time()
            
            # Evaluate population
            scores = evaluate_population(pop, conn_map, N, a, b, vreset, d, k, vr, vt, vpeak, C, E,
                                       alpha, cue_wave, go_wave, crit_Exp, crit_Cont, 
                                       crit_indices, pass_ids, TMAX, False)
            
            # Store all DNA and scores from this generation
            for i, (idx, exp, cont) in enumerate(scores):
                dna_record = {
                    'process_id': process_id,
                    'generation': gen,
                    'individual_id': i,
                    'dna': pop[i].copy(),
                    'exp_score': int(exp),
                    'cont_score': int(cont),
                    'total_score': int(exp + cont),
                    'timestamp': time.time()
                }
                all_tested_dna.append(dna_record)
                all_scores.append(int(exp + cont))
            
            # Create population records for next generation
            dpops = [{"dna": pop[i], "dna_score": int(exp+cont)} 
                    for i, (idx, exp, cont) in enumerate(scores)]
            
            # Generate next population (except for last generation)
            if gen < num_generations - 1:
                pop = spawn_next_population(dpops, cfg)
            
            gen_end = time.time()
            
            # Statistics for this generation
            gen_scores = [int(exp + cont) for _, exp, cont in scores]
            gen_stats = {
                'generation': gen,
                'process_id': process_id,
                'best_score': max(gen_scores),
                'mean_score': np.mean(gen_scores),
                'std_score': np.std(gen_scores),
                'time_taken': gen_end - gen_start,
                'population_size': len(scores)
            }
            generation_stats.append(gen_stats)
            
            # Update shared counter and print progress
            with lock:
                shared_counter.value += 1
                total_progress = shared_counter.value
            
            print(f"Process {process_id}: Gen {gen+1}/{num_generations} completed. "
                  f"Best: {max(gen_scores)}, Mean: {np.mean(gen_scores):.1f}, "
                  f"Time: {gen_end - gen_start:.2f}s "
                  f"(Total progress: {total_progress} generations)")
        
        # Save results for this process
        process_results = {
            'process_id': process_id,
            'config_name': config_name,
            'num_generations': num_generations,
            'all_tested_dna': all_tested_dna,
            'generation_stats': generation_stats,
            'final_population': [{"dna": dna, "dna_score": score} for dna, score in zip(pop, all_scores[-len(pop):])],
            'total_individuals_tested': len(all_tested_dna),
            'best_overall_score': max(all_scores),
            'completion_time': time.time()
        }
        
        # Save to file
        process_file = results_dir / f"process_{process_id}_results.pkl"
        with open(process_file, 'wb') as f:
            pickle.dump(process_results, f)
        
        print(f"Process {process_id}: Completed! Tested {len(all_tested_dna)} individuals. "
              f"Best score: {max(all_scores)}. Results saved to {process_file}")
        
        return process_results
        
    except Exception as e:
        error_msg = f"Process {process_id} failed with error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return {'process_id': process_id, 'error': str(e), 'traceback': traceback.format_exc()}


# ============================================================================
# Main execution function
# ============================================================================

def run_multiprocess_ga(config_name: str = "medium", num_processes: Optional[int] = None, 
                       num_generations: Optional[int] = None, results_dir: Optional[str] = None) -> Dict:
    """
    Run genetic algorithm using multiprocessing.
    
    Args:
        config_name: Configuration name from GA_CONFIG (default: "medium")
        num_processes: Number of processes (default: CPU count)
        num_generations: Number of generations per process (default: from config)
        results_dir: Results directory (default: timestamped dir under ./results)
    
    Returns:
        Dictionary with aggregated results from all processes
    """
    
    start_time = time.time()
    
    # Set defaults
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    cfg = GA_CONFIG[config_name]
    if num_generations is None:
        num_generations = cfg["NUM_GENERATIONS"]
    
    # Create results directory
    if results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = Path(f"results/multiprocess_ga_{config_name}_{timestamp}")
    else:
        results_path = Path(results_dir)
    
    results_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("MULTIPROCESSING GENETIC ALGORITHM")
    print("=" * 80)
    print(f"Configuration: {config_name}")
    print(f"Processes: {num_processes}")
    print(f"Generations per process: {num_generations}")
    print(f"Population per process: {cfg['POP_SIZE']}")
    total_individuals = num_processes * cfg['POP_SIZE'] * num_generations
    print(f"Total individuals to test: {total_individuals:,}")
    print(f"Results directory: {results_path}")
    print(f"Expected total generations: {num_processes * num_generations}")
    print("=" * 80)
    
    # Create shared counter for progress tracking
    manager = mp.Manager()
    shared_counter = manager.Value('i', 0)
    lock = manager.Lock()
    
    # Create worker arguments
    worker_args = [
        (proc_id, config_name, num_generations, proc_id * 1000, results_path, shared_counter, lock)
        for proc_id in range(num_processes)
    ]
    
    # Start multiprocessing
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(worker_process, worker_args)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Aggregate results
    all_dna_tested = []
    all_generation_stats = []
    process_results = []
    
    best_overall_score = 0
    total_individuals = 0
    successful_processes = 0
    
    for result in results:
        if 'error' not in result:
            successful_processes += 1
            process_results.append(result)
            all_dna_tested.extend(result['all_tested_dna'])
            all_generation_stats.extend(result['generation_stats'])
            total_individuals += result['total_individuals_tested']
            best_overall_score = max(best_overall_score, result['best_overall_score'])
        else:
            print(f"Process {result['process_id']} failed: {result['error']}")
    
    # Create summary
    summary = {
        'config_name': config_name,
        'num_processes': num_processes,
        'successful_processes': successful_processes,
        'num_generations': num_generations,
        'total_runtime': total_time,
        'total_individuals_tested': total_individuals,
        'best_overall_score': best_overall_score,
        'individuals_per_second': total_individuals / total_time if total_time > 0 else 0,
        'results_directory': str(results_path),
        'start_time': start_time,
        'end_time': end_time
    }
    
    # Save aggregated results
    aggregated_results = {
        'summary': summary,
        'all_dna_tested': all_dna_tested,
        'generation_stats': all_generation_stats,
        'process_results': process_results,
        'config_used': cfg
    }
    
    summary_file = results_path / "aggregated_results.pkl"
    with open(summary_file, 'wb') as f:
        pickle.dump(aggregated_results, f)
    
    # Save human-readable summary
    summary_text = f"""
MULTIPROCESSING GENETIC ALGORITHM RESULTS
==========================================

Configuration: {config_name}
Total Runtime: {total_time:.2f} seconds ({total_time/3600:.2f} hours)
Processes: {num_processes} (successful: {successful_processes})
Generations per process: {num_generations}
Total individuals tested: {total_individuals:,}
Performance: {total_individuals / total_time:.1f} individuals/second

Best overall score: {best_overall_score}
Results saved to: {results_path}

Individual Process Performance:
"""
    
    for result in process_results:
        process_time = result['completion_time'] - start_time
        individuals_tested = result['total_individuals_tested']
        summary_text += f"  Process {result['process_id']}: {individuals_tested:,} individuals, "
        summary_text += f"best score {result['best_overall_score']}, "
        summary_text += f"{individuals_tested/process_time:.1f} ind/sec\\n"
    
    with open(results_path / "summary.txt", 'w') as f:
        f.write(summary_text)
    
    print("\n" + "=" * 80)
    print("MULTIPROCESSING GA COMPLETED!")
    print("=" * 80)
    print(f"Total runtime: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
    print(f"Total individuals tested: {total_individuals:,}")
    print(f"Performance: {total_individuals / total_time:.1f} individuals/second")
    print(f"Best overall score: {best_overall_score}")
    print(f"Results saved to: {results_path}")
    print(f"Summary file: {summary_file}")
    print("=" * 80)
    
    return aggregated_results


# ============================================================================
# Command line interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run multiprocessing genetic algorithm")
    parser.add_argument("--config", choices=GA_CONFIG.keys(), default="medium",
                       help="GA configuration to use")
    parser.add_argument("--processes", type=int, default=None,
                       help="Number of processes (default: CPU count)")
    parser.add_argument("--generations", type=int, default=None,
                       help="Number of generations per process (default: from config)")
    parser.add_argument("--results-dir", type=str, default=None,
                       help="Results directory (default: timestamped dir)")
    
    args = parser.parse_args()
    
    # Run the multiprocessing GA
    results = run_multiprocess_ga(
        config_name=args.config,
        num_processes=args.processes,
        num_generations=args.generations,
        results_dir=args.results_dir
    )