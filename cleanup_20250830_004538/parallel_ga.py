"""
Parallel Genetic Algorithm with Complete Data Saving

IMPORTANT SETUP NOTES:
======================

This standalone script requires your GA functions to be available.
You have several options:

1. RECOMMENDED: Use the Jupyter notebook version (v2.ipynb) instead
   - The notebook version has all functions integrated
   - Works seamlessly with multiprocessing

2. Import your GA functions:
   - Add: from your_module import evaluate_population, spawn_next_population
   - Replace "your_module" with your actual module name

3. Copy function definitions:
   - Copy your evaluate_population and spawn_next_population functions
   - Paste them into this file before the worker function

Usage Examples:
   python parallel_ga.py small 4 10
   python parallel_ga.py F 8 50
"""

import os
# Set these before importing numba-compiled modules
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_DISABLE_JIT"] = "0"

import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from time import perf_counter
from datetime import datetime
import sys
import pickle
import json
import sqlite3
import pandas as pd
import glob

# Import your existing modules
from src.constants import *

# TODO: Import your GA functions here
# from your_module import evaluate_population, spawn_next_population

# Your existing functions (you would import these from your modules)
# For demonstration, I'm including simplified versions here

def initialize_population(size, upper_bound, synapses, inhibited):
    """Create initial population with proper int32 data types."""
    base_vector = np.ones(len(synapses), dtype=np.int32)
    for idx, conn in enumerate(synapses):
        if conn[0] in inhibited:
            base_vector[idx] *= -1
    
    vectors = []
    while len(vectors) < size:
        random_values = np.random.randint(0, upper_bound, len(base_vector), dtype=np.int32)
        random_vector = (base_vector * random_values).astype(np.int32)
        vectors.append(random_vector)
    
    return np.array(vectors, dtype=np.int32)

def initialize_connection_mapping(synapses, neuron_names):
    """Create connection mapping for synapses."""
    indices = np.zeros((len(synapses), 2), dtype=np.int32)
    for idx, conn in enumerate(synapses):
        indices[idx, 0] = neuron_names.index(conn[0])  # pre-synaptic
        indices[idx, 1] = neuron_names.index(conn[1])  # post-synaptic
    return indices

def run_single_ga_instance(args):
    """
    Run a single genetic algorithm instance in a separate process.
    
    This function contains the complete GA loop and is designed to be
    executed by each process with different initial populations.
    """
    process_id, config_name, num_generations, random_seed, tmax, save_all_data = args
    
    # Set unique random seed for this process
    np.random.seed(random_seed)
    
    print(f"Process {process_id}: Starting GA with seed {random_seed}")
    
    # Import the heavy computation functions here to avoid issues with multiprocessing
    # IMPORTANT: You need to ensure these functions are available!
    try:
        # Try importing from your existing modules
        # Uncomment and modify these lines to match your actual module structure:
        # from your_ga_module import evaluate_population, spawn_next_population
        
        # For now, this assumes the functions are available in the global scope
        # If you get a NameError, you need to either:
        # 1. Import the functions from your modules
        # 2. Copy the function definitions into this file
        # 3. Use the Jupyter notebook version instead
        
        # Test if functions are available
        if 'evaluate_population' not in globals():
            raise NameError("evaluate_population function not found")
        if 'spawn_next_population' not in globals():
            raise NameError("spawn_next_population function not found")
            
    except NameError as e:
        print(f"❌ Error: Required function not found: {e}")
        print("💡 Solution: Either:")
        print("   1. Use the Jupyter notebook version (v2.ipynb)")
        print("   2. Import your GA functions at the top of this file")
        print("   3. Copy your GA function definitions into this file")
        raise
    
    # Initialize this process's population
    cfg = GA_CONFIG[config_name]
    pop = initialize_population(cfg["POP_SIZE"], 
                              cfg["DNA_BOUNDS"][1], 
                              ACTIVE_SYNAPSES,
                              INHIBITORY_NEURONS)
    
    conn_map = initialize_connection_mapping(ACTIVE_SYNAPSES, NEURON_NAMES)
    
    # Initialize necessary simulation parameters
    N = len(NEURON_NAMES)
    ALPHA_L = 250
    td = np.arange(1, ALPHA_L + 1, dtype=np.float32)
    alpha = (td / 30) * np.exp((30 - td) / 30)   

    cue_wave = np.zeros(tmax, dtype=np.float32)
    go_wave = np.zeros_like(cue_wave)
    cue_wave[EPOCHS['sample'][0]:EPOCHS['sample'][1]] = CUE_STRENGTH
    go_wave[EPOCHS['response'][0]:EPOCHS['response'][0] + GO_DURATION] = GO_STRENGTH

    # Create criterion matrices
    conditions = []
    for condition in CRITERIA:
        condition_criteria = []
        for neuron_name, neuron in CRITERIA[condition].items():
            idx = NEURON_NAMES.index(neuron_name)
            baseline = np.ones(tmax, np.uint8) if neuron_name in TONICALLY_ACTIVE_NEURONS else np.zeros(tmax, np.uint8)
            start = neuron["interval"][0]
            end = neuron["interval"][1]
            target_status = neuron["io"]
            
            if target_status == "off":
                baseline[start:end] = 0
            elif target_status == "on":
                baseline[start:end] = 1

            baseline = baseline.reshape(tmax//BIN_SIZE, BIN_SIZE)
            baseline = np.sum(baseline, axis=1, dtype=np.uint32)
            baseline = (baseline != 0).astype(np.uint8)

            condition_criteria.append((neuron_name, idx, baseline))
        condition_criteria = sorted(condition_criteria, key=lambda tup: tup[1])
        conditions.append(condition_criteria)

    crit_Exp, crit_Cont = conditions
    crit_indices = np.array([neu[1] for neu in crit_Cont])
    crit_Exp = np.vstack([neu[2] for neu in crit_Exp])
    crit_Cont = np.vstack([neu[2] for neu in crit_Cont])
    
    pass_ids = [NEURON_NAMES.index(x) for x in ["VMresp", "ALMresp", "SNR3"]]
    pass_ids = np.array(pass_ids)
    
    # Track the best individual across generations
    best_individual = None
    best_score = -1
    generation_scores = []
    
    # NEW: Store ALL DNA vectors and scores if requested
    all_individuals = [] if save_all_data else None
    
    start_time = perf_counter()
    
    # Main GA loop
    for gen in range(num_generations):
        # Evaluate population
        scores = evaluate_population(pop,
                                   conn_map, N,                                    
                                   a, b, vreset, d, k, vr, vt, vpeak, C, E,
                                   alpha, cue_wave, go_wave,
                                   crit_Exp, crit_Cont, crit_indices, pass_ids,
                                   tmax, False)
        
        # Create population records
        dpops = [{"dna": pop[i], "dna_score": exp+cont} for i, (idx, exp, cont) in enumerate(scores)]
        
        # NEW: Save all individuals from this generation
        if save_all_data:
            for i, (idx, exp_score, cont_score) in enumerate(scores):
                individual_record = {
                    'process_id': process_id,
                    'generation': gen,
                    'individual_id': i,
                    'dna': pop[i].copy(),  # Copy to avoid reference issues
                    'exp_score': int(exp_score),
                    'cont_score': int(cont_score),
                    'total_score': int(exp_score + cont_score),
                    'random_seed': random_seed,
                    'config_name': config_name,
                    'timestamp': datetime.now().isoformat()
                }
                all_individuals.append(individual_record)
        
        # Track best individual
        current_best = max(dpops, key=lambda x: x["dna_score"])
        if current_best["dna_score"] > best_score:
            best_score = current_best["dna_score"]
            best_individual = current_best["dna"].copy()
        
        # Store generation statistics
        scores_list = [record["dna_score"] for record in dpops]
        generation_scores.append({
            'generation': gen,
            'best': max(scores_list),
            'mean': np.mean(scores_list),
            'std': np.std(scores_list)
        })
        
        # Print progress
        if gen % 5 == 0 or gen == num_generations - 1:
            print(f"Process {process_id}: Gen {gen:3d}, Best: {max(scores_list):4d}, Mean: {np.mean(scores_list):6.1f}")
        
        # Generate next population (except for last generation)
        if gen < num_generations - 1:
            pop = spawn_next_population(dpops, cfg)
    
    end_time = perf_counter()
    runtime = end_time - start_time
    
    print(f"Process {process_id}: Completed in {runtime:.2f}s, Final best: {best_score}")
    
    result = {
        'process_id': process_id,
        'random_seed': random_seed,
        'best_individual': best_individual,
        'best_score': best_score,
        'generation_scores': generation_scores,
        'runtime': runtime,
        'final_population_size': len(pop),
        'config_name': config_name
    }
    
    # Add all individuals if requested
    if save_all_data:
        result['all_individuals'] = all_individuals
        print(f"Process {process_id}: Saved {len(all_individuals)} individuals")
    
    return result

def run_parallel_ga(config_name="small", num_processes=None, num_generations=None, 
                    tmax=5000, save_all_data=True, output_dir="results"):
    """
    Run multiple GA instances in parallel using multiprocessing.
    
    This is the main function that coordinates parallel execution.
    """
    # Setup multiprocessing method
    setup_multiprocessing()
    
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    cfg = GA_CONFIG[config_name]
    if num_generations is None:
        num_generations = cfg["NUM_GENERATIONS"]
    
    print(f"Starting {num_processes} parallel GA processes")
    print(f"Config: {config_name}, Generations: {num_generations}, Population size: {cfg['POP_SIZE']}")
    print(f"Total individuals to evaluate: {num_processes * cfg['POP_SIZE'] * num_generations}")
    print("-" * 60)
    
    # Create arguments for each process with different random seeds
    base_seed = int(datetime.now().timestamp())
    process_args = []
    for i in range(num_processes):
        seed = base_seed + i * 1000  # Ensure seeds are well separated
        process_args.append((i, config_name, num_generations, seed, tmax, save_all_data))
    
    start_time = perf_counter()
    
    # Run processes in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(run_single_ga_instance, process_args)
    
    end_time = perf_counter()
    total_runtime = end_time - start_time
    
    print("\n" + "="*60)
    print(f"All processes completed in {total_runtime:.2f}s")
    
    # Find overall best result
    best_result = max(results, key=lambda x: x['best_score'])
    print(f"Overall best score: {best_result['best_score']} (Process {best_result['process_id']})")
    
    # Save results if data was collected
    file_paths = {}
    if save_all_data:
        file_paths = save_results_to_files(results, output_dir, config_name, save_all_data)
    
    return results, total_runtime, file_paths

def save_results_to_files(results, output_dir, config_name, save_all_data):
    """Save results to multiple file formats for analysis."""
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{output_dir}/parallel_ga_{config_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    file_paths = {}
    
    # 1. Save complete results as pickle
    pickle_path = os.path.join(run_dir, "complete_results.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    file_paths['pickle'] = pickle_path
    print(f"✅ Saved complete results to: {pickle_path}")
    
    # 2. Save summary as JSON
    summary_data = {
        'config_name': config_name,
        'num_processes': len(results),
        'timestamp': datetime.now().isoformat(),
        'total_individuals': sum(len(r.get('all_individuals', [])) for r in results),
        'best_scores': [r['best_score'] for r in results],
        'process_summary': [
            {
                'process_id': r['process_id'],
                'best_score': r['best_score'],
                'runtime': r['runtime'],
                'random_seed': r['random_seed']
            } for r in results
        ]
    }
    
    json_path = os.path.join(run_dir, "summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    file_paths['json'] = json_path
    print(f"✅ Saved summary to: {json_path}")
    
    # 3. Save all individuals to CSV and database if data was collected
    if save_all_data and any('all_individuals' in r for r in results):
        # Collect all individuals
        all_individuals = []
        for result in results:
            if 'all_individuals' in result:
                all_individuals.extend(result['all_individuals'])
        
        if all_individuals:
            # Convert to DataFrame
            df_data = []
            for ind in all_individuals:
                row = {
                    'process_id': ind['process_id'],
                    'generation': ind['generation'],
                    'individual_id': ind['individual_id'],
                    'exp_score': ind['exp_score'],
                    'cont_score': ind['cont_score'],
                    'total_score': ind['total_score'],
                    'random_seed': ind['random_seed'],
                    'config_name': ind['config_name'],
                    'timestamp': ind['timestamp']
                }
                # Add DNA genes as separate columns
                for i, gene in enumerate(ind['dna']):
                    row[f'gene_{i:02d}'] = int(gene)
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            
            # Save as CSV
            csv_path = os.path.join(run_dir, "all_individuals.csv")
            df.to_csv(csv_path, index=False)
            file_paths['csv'] = csv_path
            print(f"✅ Saved {len(df)} individuals to CSV: {csv_path}")
            
            # Save to SQLite database
            db_path = os.path.join(run_dir, "ga_results.db")
            conn = sqlite3.connect(db_path)
            df.to_sql('individuals', conn, if_exists='replace', index=False)
            conn.close()
            file_paths['database'] = db_path
            print(f"✅ Saved to SQLite database: {db_path}")
    
    return file_paths

def analyze_results(results, total_runtime):
    """Analyze and display results from parallel GA runs."""
    print("\n" + "="*80)
    print("PARALLEL GA RESULTS ANALYSIS")
    print("="*80)
    
    # Overall statistics
    best_scores = [r['best_score'] for r in results]
    runtimes = [r['runtime'] for r in results]
    
    print(f"Number of processes: {len(results)}")
    print(f"Total runtime: {total_runtime:.2f}s")
    print(f"Average process runtime: {np.mean(runtimes):.2f}s")
    print(f"Process runtime std: {np.std(runtimes):.2f}s")
    
    print(f"\nBest scores across processes:")
    print(f"  Best: {max(best_scores)}")
    print(f"  Mean: {np.mean(best_scores):.2f}")
    print(f"  Std:  {np.std(best_scores):.2f}")
    print(f"  Min:  {min(best_scores)}")
    
    # Per-process summary
    print(f"\nPer-process results:")
    print("Process | Best Score | Runtime(s)")
    print("-" * 35)
    for r in sorted(results, key=lambda x: x['best_score'], reverse=True):
        print(f"   {r['process_id']:2d}   |    {r['best_score']:4d}    |   {r['runtime']:6.2f}")
    
    # Find the overall best individual
    best_result = max(results, key=lambda x: x['best_score'])
    print(f"\nBest individual found:")
    print(f"  Process: {best_result['process_id']}")
    print(f"  Score: {best_result['best_score']}")
    print(f"  DNA (first 10 genes): {best_result['best_individual'][:10]}")
    
    return best_result

def main():
    """Main function for command-line execution."""
    print("Parallel Genetic Algorithm Demo")
    print("===============================")
    
    # You can modify these parameters
    config = "small"  # or "medium", "F", etc.
    processes = 4
    generations = 10
    tmax = 2000  # Shorter for demo
    
    if len(sys.argv) > 1:
        config = sys.argv[1]
    if len(sys.argv) > 2:
        processes = int(sys.argv[2])
    if len(sys.argv) > 3:
        generations = int(sys.argv[3])
    
    print(f"Configuration: {config}")
    print(f"Processes: {processes}")
    print(f"Generations: {generations}")
    print(f"tmax: {tmax}")
    print()
    
    # Run parallel GA
    results, runtime, file_paths = run_parallel_ga(
        config_name=config,
        num_processes=processes,
        num_generations=generations,
        tmax=tmax,
        save_all_data=True
    )
    
    # Analyze results
    best_result = analyze_results(results, runtime)
    
    # Show saved files
    if file_paths:
        print(f"\n📁 Results saved:")
        for format_type, path in file_paths.items():
            print(f"   {format_type}: {path}")
    
    print(f"\n🎉 Parallel GA completed successfully!")
    print(f"Best individual found has score: {best_result['best_score']}")
    
    return results, best_result

def setup_multiprocessing():
    """Setup multiprocessing to work correctly across platforms"""
    try:
        # Try to use fork method (better for most use cases)
        current_method = mp.get_start_method(allow_none=True)
        if current_method is None:
            try:
                mp.set_start_method('fork')
                print("✅ Using 'fork' multiprocessing method")
            except RuntimeError:
                try:
                    mp.set_start_method('spawn')
                    print("✅ Using 'spawn' multiprocessing method")
                except RuntimeError:
                    print(f"ℹ️  Using default multiprocessing method: {mp.get_start_method()}")
        else:
            print(f"ℹ️  Using multiprocessing method: {current_method}")
    except Exception as e:
        print(f"⚠️  Multiprocessing setup warning: {e}")

if __name__ == "__main__":
    # Setup multiprocessing correctly
    setup_multiprocessing()
    results, best = main() 