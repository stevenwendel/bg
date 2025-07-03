"""
Numba Performance Test for Notebook

Copy and paste these functions into your v2.ipynb notebook to test Numba performance.
"""

import numpy as np
import time
from time import perf_counter
import os
import numba
import multiprocessing as mp

def test_numba_performance(num_runs=10, warmup_runs=2):
    """
    Test Numba performance with different configurations.
    
    This function will test:
    1. Current configuration (baseline)
    2. JIT disabled
    3. JIT enabled
    4. Different thread counts (1, 2, 4, 8, auto)
    
    Parameters:
    -----------
    num_runs : int
        Number of timed runs for each configuration
    warmup_runs : int
        Number of warmup runs (important for Numba compilation)
    """
    
    print("Numba Performance Analysis")
    print("=" * 60)
    
    # Print system info
    print(f"CPU cores: {mp.cpu_count()}")
    print(f"Numba version: {numba.__version__}")
    print(f"Current Numba threads: {numba.get_num_threads()}")
    print(f"Current JIT enabled: {not numba.config.DISABLE_JIT}")
    
    def run_timing_test(test_name, warmup_count=2):
        """Helper function to run timing test"""
        print(f"\n{'='*50}")
        print(f"Testing: {test_name}")
        print(f"{'='*50}")
        
        # Warmup runs
        print("Warming up...")
        for _ in range(warmup_count):
            experimental_score, _ = simulate(W, cue_wave, go_wave, tmax=TMAX, control=False)
            control_score, _ = simulate(W, cue_wave, go_wave, tmax=TMAX, control=True)
        
        # Actual timing runs
        times = []
        print(f"Running {num_runs} timed iterations...")
        
        for i in range(num_runs):
            start = perf_counter()
            experimental_score, _ = simulate(W, cue_wave, go_wave, tmax=TMAX, control=False)
            control_score, _ = simulate(W, cue_wave, go_wave, tmax=TMAX, control=True)
            end = perf_counter()
            
            time_taken = end - start
            times.append(time_taken)
            print(f"  Run {i+1}: {time_taken:.4f}s")
        
        # Statistics
        times = np.array(times)
        print(f"\nResults for {test_name}:")
        print(f"  Mean time: {times.mean():.4f}s ± {times.std():.4f}s")
        print(f"  Min time:  {times.min():.4f}s")
        print(f"  Max time:  {times.max():.4f}s")
        
        return times.mean()
    
    # Store results
    results = {}
    
    # Test 1: Current configuration (baseline)
    print("\n1. Testing Current Configuration (baseline)...")
    results['baseline'] = run_timing_test("Current Configuration", warmup_count=warmup_runs)
    
    # Test 2: JIT disabled
    print("\n2. Testing with JIT Disabled...")
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    numba.config.DISABLE_JIT = True
    results['jit_disabled'] = run_timing_test("JIT Disabled", warmup_count=0)
    
    # Test 3: JIT enabled (reset)
    print("\n3. Testing with JIT Enabled...")
    os.environ["NUMBA_DISABLE_JIT"] = "0"
    numba.config.DISABLE_JIT = False
    results['jit_enabled'] = run_timing_test("JIT Enabled", warmup_count=warmup_runs)
    
    # Test 4: Different thread counts
    thread_counts = [1, 2, 4, 8]
    cpu_count = mp.cpu_count()
    if cpu_count > 8:
        thread_counts.append(cpu_count)
    
    print(f"\n4. Testing Different Thread Counts (CPU cores: {cpu_count})...")
    
    for num_threads in thread_counts:
        print(f"\n   Testing with {num_threads} threads...")
        numba.set_num_threads(num_threads)
        results[f'{num_threads}_threads'] = run_timing_test(f"{num_threads} threads", warmup_count=warmup_runs)
    
    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    baseline = results['baseline']
    print(f"Baseline (current config): {baseline:.4f}s")
    print(f"JIT Disabled: {results['jit_disabled']:.4f}s (speedup: {results['jit_disabled']/baseline:.2f}x)")
    print(f"JIT Enabled: {results['jit_enabled']:.4f}s (speedup: {results['jit_enabled']/baseline:.2f}x)")
    
    print(f"\nThreading Results:")
    for key, time_taken in results.items():
        if 'threads' in key:
            speedup = baseline / time_taken
            print(f"  {key}: {time_taken:.4f}s (speedup: {speedup:.2f}x)")
    
    print("\nKey Insights:")
    print("1. JIT compilation adds overhead on first run but speeds up subsequent runs")
    print("2. More threads doesn't always mean faster execution (overhead vs benefit)")
    print("3. The optimal configuration depends on your specific workload and hardware")
    print("4. For small workloads, single-threaded might be fastest due to threading overhead")
    
    return results

# Example usage in notebook:
# results = test_numba_performance(num_runs=5, warmup_runs=2)

def quick_numba_test():
    """
    Quick test to see immediate Numba effects.
    Run this to see the difference between JIT enabled/disabled.
    """
    print("Quick Numba Test")
    print("=" * 40)
    
    # Test with JIT enabled
    print("\n1. Testing with JIT enabled...")
    os.environ["NUMBA_DISABLE_JIT"] = "0"
    numba.config.DISABLE_JIT = False
    
    start = perf_counter()
    for i in range(5):
        experimental_score, _ = simulate(W, cue_wave, go_wave, tmax=TMAX, control=False)
        control_score, _ = simulate(W, cue_wave, go_wave, tmax=TMAX, control=True)
    end = perf_counter()
    jit_enabled_time = (end - start) / 5
    print(f"JIT Enabled: {jit_enabled_time:.4f}s per run")
    
    # Test with JIT disabled
    print("\n2. Testing with JIT disabled...")
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    numba.config.DISABLE_JIT = True
    
    start = perf_counter()
    for i in range(5):
        experimental_score, _ = simulate(W, cue_wave, go_wave, tmax=TMAX, control=False)
        control_score, _ = simulate(W, cue_wave, go_wave, tmax=TMAX, control=True)
    end = perf_counter()
    jit_disabled_time = (end - start) / 5
    print(f"JIT Disabled: {jit_disabled_time:.4f}s per run")
    
    # Reset to enabled
    os.environ["NUMBA_DISABLE_JIT"] = "0"
    numba.config.DISABLE_JIT = False
    
    print(f"\nSpeedup with JIT: {jit_disabled_time/jit_enabled_time:.2f}x")

# Example usage in notebook:
# quick_numba_test() 