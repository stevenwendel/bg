"""
Simple Numba Performance Test

This script modifies your existing timing code to test different Numba configurations.
Run this to see how JIT compilation and threading affect your simulation performance.
"""

import numpy as np
import time
from time import perf_counter
import os
import numba
import multiprocessing as mp

# Import your existing simulation setup
from src.constants import *
from v2 import simulate, step_kernel, W, cue_wave, go_wave, alpha

def test_timing_with_config(test_name, num_runs=10, warmup_runs=2):
    """Test timing with current Numba configuration"""
    print(f"\n{'='*50}")
    print(f"Testing: {test_name}")
    print(f"{'='*50}")
    
    # Warmup runs (important for Numba compilation)
    print("Warming up...")
    for _ in range(warmup_runs):
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

def main():
    """Run comprehensive Numba performance tests"""
    print("Numba Performance Analysis")
    print("=" * 60)
    
    # Print system info
    print(f"CPU cores: {mp.cpu_count()}")
    print(f"Numba version: {numba.__version__}")
    print(f"Current Numba threads: {numba.get_num_threads()}")
    print(f"Current JIT enabled: {not numba.config.DISABLE_JIT}")
    
    # Store baseline for comparison
    baseline_time = None
    
    # Test 1: Current configuration (baseline)
    print("\n1. Testing Current Configuration (baseline)...")
    baseline_time = test_timing_with_config("Current Configuration", num_runs=10, warmup_runs=2)
    
    # Test 2: JIT disabled
    print("\n2. Testing with JIT Disabled...")
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    numba.config.DISABLE_JIT = True
    jit_disabled_time = test_timing_with_config("JIT Disabled", num_runs=10, warmup_runs=0)
    
    # Test 3: JIT enabled (reset)
    print("\n3. Testing with JIT Enabled...")
    os.environ["NUMBA_DISABLE_JIT"] = "0"
    numba.config.DISABLE_JIT = False
    jit_enabled_time = test_timing_with_config("JIT Enabled", num_runs=10, warmup_runs=2)
    
    # Test 4: Different thread counts
    thread_counts = [1, 2, 4, 8]
    cpu_count = mp.cpu_count()
    if cpu_count > 8:
        thread_counts.append(cpu_count)
    
    print(f"\n4. Testing Different Thread Counts (CPU cores: {cpu_count})...")
    thread_results = {}
    
    for num_threads in thread_counts:
        print(f"\n   Testing with {num_threads} threads...")
        numba.set_num_threads(num_threads)
        thread_results[num_threads] = test_timing_with_config(f"{num_threads} threads", num_runs=10, warmup_runs=2)
    
    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    if baseline_time:
        print(f"Baseline (current config): {baseline_time:.4f}s")
        print(f"JIT Disabled: {jit_disabled_time:.4f}s (speedup: {jit_disabled_time/baseline_time:.2f}x)")
        print(f"JIT Enabled: {jit_enabled_time:.4f}s (speedup: {jit_enabled_time/baseline_time:.2f}x)")
        
        print(f"\nThreading Results:")
        for threads, time_taken in thread_results.items():
            speedup = baseline_time / time_taken
            print(f"  {threads} threads: {time_taken:.4f}s (speedup: {speedup:.2f}x)")
    
    print("\nKey Insights:")
    print("1. JIT compilation adds overhead on first run but speeds up subsequent runs")
    print("2. More threads doesn't always mean faster execution (overhead vs benefit)")
    print("3. The optimal configuration depends on your specific workload and hardware")
    print("4. For small workloads, single-threaded might be fastest due to threading overhead")

if __name__ == "__main__":
    main() 