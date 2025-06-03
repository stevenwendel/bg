import itertools
import numpy as np
from datetime import datetime
import os
import pickle
import sys
import random
import time
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from multiprocessing import freeze_support

src_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(src_path)

from src.constants import GA_CONFIG
from main import main

def run_single_optimization(params, results_dir):
    """Run a single optimization with given parameters and save results."""
    try:
        # Update GA_CONFIG with new parameters
        GA_CONFIG['E'].update(params)
        
        # Run the GA
        main()
        
        # Load the results from the latest run
        latest_file = max([f for f in os.listdir('./data') if f.startswith('E_')], 
                        key=lambda x: os.path.getctime(os.path.join('./data', x)))
        
        with open(os.path.join('./data', latest_file), 'rb') as f:
            run_data = pickle.load(f)
        
        result = {
            'parameters': params,
            'best_score': run_data['best_score']
        }
        
        # Save intermediate results
        results_file = os.path.join(results_dir, 'optimization_results.pkl')
        if os.path.exists(results_file):
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
        else:
            results = []
        results.append(result)
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
            
        return result
        
    except Exception as e:
        print(f"Error in optimization: {e}")
        return None

def random_search(num_samples=20):
    """Perform random search optimization."""
    # Create directory for results
    results_dir = f'./data/random_search_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Define parameter ranges
    mut_rates = [0.2, 0.35, 0.5, 0.65]
    mut_sigmas = [0.2, 0.35, 0.5, 0.65]
    elite_sizes = [1, 5, 10, 20]
    dna_bounds = [[0, 250], [0, 500], [0, 1000]]
    pop_gen_combinations = [
        (50, 600), (100, 300), (200, 150), (300, 100)
    ]
    
    results = []
    for i in range(num_samples):
        params = {
            'MUT_RATE': random.choice(mut_rates),
            'MUT_SIGMA': random.choice(mut_sigmas),
            'ELITE_SIZE': random.choice(elite_sizes),
            'DNA_BOUNDS': random.choice(dna_bounds),
            'POP_SIZE': None,
            'NUM_GENERATIONS': None
        }
        pop_size, num_generations = random.choice(pop_gen_combinations)
        params['POP_SIZE'] = pop_size
        params['NUM_GENERATIONS'] = num_generations

        print(f"\nRandom search {i+1}/{num_samples}: {params}")
        result = run_single_optimization(params, results_dir)
        if result:
            results.append(result)
    
    analyze_results(results, results_dir)
    return results

def bayesian_optimization(n_calls=20):
    """Perform Bayesian optimization using scikit-optimize."""
    # Create directory for results
    results_dir = f'./data/bayesian_opt_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    os.makedirs(results_dir, exist_ok=True)
    max_simulations = 5000

    def objective(params):
        mut_rate, mut_sigma, elite_size, dna_bound_upper, pop_size = params
        num_generations = max_simulations//pop_size
        
        config = {
            'MUT_RATE': mut_rate,
            'MUT_SIGMA': mut_sigma,
            'ELITE_SIZE': elite_size,
            'DNA_BOUNDS': [0, dna_bound_upper],
            'POP_SIZE': pop_size,
            'NUM_GENERATIONS': num_generations
        }
        
        # Print current GA parameters
        print("\nCurrent GA Parameters:")
        print("=====================")
        for key, value in config.items():
            print(f"{key}: {value}")
        print("=====================\n")
        
        result = run_single_optimization(config, results_dir)

        if result:
            return -result['best_score']  # Negative because skopt minimizes
        return 1e6  # Large penalty for failed runs

    # Define search space
    search_space = [
        Real(0.1, 0.8, name='mut_rate'),
        Real(0.1, 0.8, name='mut_sigma'),
        Integer(0, 20, name='elite_size'),
        Integer(300, 600, name='dna_bound_upper'),
        Integer(50, 500, name='pop_size')
    ]

    # Run optimization
    result = gp_minimize(
        objective, 
        search_space, 
        n_calls=n_calls, 
        random_state=41,
        verbose=True
    )
    
    # Convert results to our format
    results = []
    for i, (params, score) in enumerate(zip(result.x_iters, -result.func_vals)):
        mut_rate, mut_sigma, elite_size, dna_bound_upper, pop_size = params
        num_generations = max_simulations//pop_size
        
        results.append({
            'parameters': {
                'MUT_RATE': mut_rate,
                'MUT_SIGMA': mut_sigma,
                'ELITE_SIZE': elite_size,
                'DNA_BOUNDS': [0, dna_bound_upper],
                'POP_SIZE': pop_size,
                'NUM_GENERATIONS': num_generations
            },
            'best_score': score
        })
    
    analyze_results(results, results_dir)
    return results

def analyze_results(results, results_dir):
    """Analyze and save optimization results."""
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    # Basic statistics
    print("\nTop 10 Best Performing Combinations:")
    print(df.nlargest(10, 'best_score')[['parameters', 'best_score']])
    
    # Save detailed analysis
    with open(os.path.join(results_dir, 'analysis.txt'), 'w') as f:
        f.write("Genetic Algorithm Optimization Results\n")
        f.write("===================================\n\n")
        
        f.write("Top 10 Best Performing Combinations:\n")
        f.write(df.nlargest(10, 'best_score').to_string())
        f.write("\n\n")
        
        f.write("Parameter Impact Analysis:\n")
        for param in ['MUT_RATE', 'MUT_SIGMA', 'ELITE_SIZE', 'POP_SIZE']:
            f.write(f"\n{param} Analysis:\n")
            # Extract parameter values from the nested dictionary
            param_values = df['parameters'].apply(lambda x: x[param])
            param_stats = df.groupby(param_values)['best_score'].agg(['mean', 'std', 'max'])
            f.write(param_stats.to_string())
            f.write("\n")

if __name__ == "__main__":
    optimizer_start_time = time.time()
    freeze_support()  # Required for multiprocessing on Windows/macOS
    
    # Choose which optimization method to run
    method = "bayesian"  # or "random"
    
    if method == "random":
        print("Running random search optimization...")
        results = random_search(num_samples=20)
    else:
        print("Running Bayesian optimization...")
        results = bayesian_optimization(n_calls=50) 
        
    optimizer_end_time = time.time()
    optimizer_duration = (optimizer_end_time - optimizer_start_time)//60
    print(f"Optimizer algorithm took {optimizer_duration} minutes.")