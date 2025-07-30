"""
Comprehensive Numba Performance Test

This script tests the impact of:
1. Numba JIT compilation (enabled vs disabled)
2. Number of threads (1, 2, 4, 8, auto)
3. Different Numba optimization flags

Educational notes:
- Numba JIT compilation has overhead on first run (compilation time)
- Subsequent runs benefit from cached compiled code
- Threading performance depends on your CPU cores and workload
- Some operations may not benefit from parallelization due to overhead
"""

import numpy as np
import time
from time import perf_counter
import os
import numba
from numba import njit, prange, float32, uint8
import multiprocessing as mp

# Import your simulation components
from src.constants import *
from v2 import simulate, step_kernel, W, cue_wave, go_wave, alpha

def create_non_numba_step_kernel():
    """Create a pure Python version of step_kernel for comparison"""
    def step_kernel_python(V, U, Ibuf, t_ptr,
                          a, b, vreset, d, k, vr, vt, vpeak, C, E, 
                          W, alpha):
        n, L = V.size, alpha.size
        spk = np.zeros(n, dtype=np.uint8)

        # integrate -------------------------------------------------------
        for i in range(n):  # No prange - pure Python loop
            I = Ibuf[i, t_ptr]
            dV = (k[i]*(V[i]-vr[i])*(V[i]-vt[i]) - U[i] + I + E[i]) / C[i]
            dU = a[i]*(b[i]*(V[i]-vr[i]) - U[i])
            V[i] += dV
            U[i] += dU
            if V[i] >= vpeak[i]:
                V[i] = vreset[i]
                U[i] += d[i]
                spk[i] = 1

        # distribute PSC --------------------------------------------------
        if spk.any():
            post_I = spk.astype(float32) @ W
            t_next = (t_ptr + 1) % L
            for k_shift in range(L):
                Ibuf[:, (t_next + k_shift) % L] += post_I * alpha[k_shift]

        Ibuf[:, t_ptr] = 0.0
        return spk, (t_ptr + 1) % L
    
    return step_kernel_python

def create_simulate_with_kernel(step_kernel_func):
    """Create a simulate function that uses the specified step kernel"""
    def simulate_with_kernel(W, cue_wave, go_wave, tmax=TMAX, control=False, return_full=False):
        N = len(NEURON_NAMES)
        V = np.full(N, -60.0, np.float32)
        U = np.zeros_like(V)
        Ibuf = np.zeros((N, 250), np.float32)  # ALPHA_L = 250
        HIST = np.zeros((N, BIN_SIZE), np.uint8)

        if return_full:
            temp_full_hist = np.zeros((N, TMAX), np.uint8)

        score = 0
        t_ptr = 0
        bin = 0

        for t in range(TMAX):
            # inject cue/go as plain current
            if control == False:
                Ibuf[0, t_ptr] += cue_wave[t]
            Ibuf[7, t_ptr] += go_wave[t]

            spk, t_ptr = step_kernel_func(V, U, Ibuf, t_ptr,
                                         a, b, vreset, d, k, vr, vt, vpeak, C, E, 
                                         W, alpha)

            if return_full:
                temp_full_hist[:, t] = spk

            cidx = t % BIN_SIZE
            HIST[:, cidx] = spk

            if cidx == BIN_SIZE - 1:
                # Scoring logic here (simplified)
                bin += 1

        if return_full:
            return score, temp_full_hist
        return score, None
    
    return simulate_with_kernel

def run_performance_test(test_name, step_kernel_func, num_runs=10, warmup_runs=2):
    """Run performance test with given kernel function"""
    print(f"\n{'='*60}")
    print(f"Testing: {test_name}")
    print(f"{'='*60}")
    
    # Warmup runs (to compile Numba functions)
    print("Warming up...")
    for _ in range(warmup_runs):
        simulate_with_kernel = create_simulate_with_kernel(step_kernel_func)
        simulate_with_kernel(W, cue_wave, go_wave, tmax=TMAX, control=False)
    
    # Actual timing runs
    times = []
    print(f"Running {num_runs} timed iterations...")
    
    for i in range(num_runs):
        start = perf_counter()
        simulate_with_kernel = create_simulate_with_kernel(step_kernel_func)
        experimental_score, _ = simulate_with_kernel(W, cue_wave, go_wave, tmax=TMAX, control=False)
        control_score, _ = simulate_with_kernel(W, cue_wave, go_wave, tmax=TMAX, control=True)
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
    print(f"  Speedup vs Python: {times.mean() / python_baseline:.2f}x")
    
    return times.mean()

def test_numba_configurations():
    """Test different Numba configurations"""
    global python_baseline
    
    print("Numba Performance Analysis")
    print("=" * 60)
    
    # Test 1: Pure Python (baseline)
    print("\n1. Testing Pure Python (baseline)...")
    step_kernel_python = create_non_numba_step_kernel()
    python_baseline = run_performance_test("Pure Python", step_kernel_python, num_runs=5, warmup_runs=0)
    
    # Test 2: Numba JIT disabled
    print("\n2. Testing Numba JIT Disabled...")
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    numba.config.DISABLE_JIT = True
    run_performance_test("Numba JIT Disabled", step_kernel, num_runs=10, warmup_runs=0)
    
    # Test 3: Numba JIT enabled (default)
    print("\n3. Testing Numba JIT Enabled (default)...")
    os.environ["NUMBA_DISABLE_JIT"] = "0"
    numba.config.DISABLE_JIT = False
    run_performance_test("Numba JIT Enabled", step_kernel, num_runs=10, warmup_runs=2)
    
    # Test 4: Different thread counts
    thread_counts = [1, 2, 4, 8]
    cpu_count = mp.cpu_count()
    if cpu_count > 8:
        thread_counts.append(cpu_count)
    
    print(f"\n4. Testing Different Thread Counts (CPU cores: {cpu_count})...")
    for num_threads in thread_counts:
        print(f"\n   Testing with {num_threads} threads...")
        numba.set_num_threads(num_threads)
        run_performance_test(f"Numba with {num_threads} threads", step_kernel, num_runs=10, warmup_runs=2)

def test_numba_flags():
    """Test different Numba optimization flags"""
    print("\n5. Testing Numba Optimization Flags...")
    
    # Create different kernel versions with different flags
    @njit(parallel=False, fastmath=False, cache=True)
    def step_kernel_no_parallel(V, U, Ibuf, t_ptr, a, b, vreset, d, k, vr, vt, vpeak, C, E, W, alpha):
        n, L = V.size, alpha.size
        spk = np.zeros(n, dtype=np.uint8)
        
        for i in range(n):  # No prange
            I = Ibuf[i, t_ptr]
            dV = (k[i]*(V[i]-vr[i])*(V[i]-vt[i]) - U[i] + I + E[i]) / C[i]
            dU = a[i]*(b[i]*(V[i]-vr[i]) - U[i])
            V[i] += dV
            U[i] += dU
            if V[i] >= vpeak[i]:
                V[i] = vreset[i]
                U[i] += d[i]
                spk[i] = 1
        
        if spk.any():
            post_I = spk.astype(float32) @ W
            t_next = (t_ptr + 1) % L
            for k_shift in range(L):
                Ibuf[:, (t_next + k_shift) % L] += post_I * alpha[k_shift]
        
        Ibuf[:, t_ptr] = 0.0
        return spk, (t_ptr + 1) % L
    
    @njit(parallel=True, fastmath=False, cache=True)
    def step_kernel_no_fastmath(V, U, Ibuf, t_ptr, a, b, vreset, d, k, vr, vt, vpeak, C, E, W, alpha):
        n, L = V.size, alpha.size
        spk = np.zeros(n, dtype=np.uint8)
        
        for i in prange(n):
            I = Ibuf[i, t_ptr]
            dV = (k[i]*(V[i]-vr[i])*(V[i]-vt[i]) - U[i] + I + E[i]) / C[i]
            dU = a[i]*(b[i]*(V[i]-vr[i]) - U[i])
            V[i] += dV
            U[i] += dU
            if V[i] >= vpeak[i]:
                V[i] = vreset[i]
                U[i] += d[i]
                spk[i] = 1
        
        if spk.any():
            post_I = spk.astype(float32) @ W
            t_next = (t_ptr + 1) % L
            for k_shift in range(L):
                Ibuf[:, (t_next + k_shift) % L] += post_I * alpha[k_shift]
        
        Ibuf[:, t_ptr] = 0.0
        return spk, (t_ptr + 1) % L
    
    # Test different configurations
    run_performance_test("Numba (no parallel, no fastmath)", step_kernel_no_parallel, num_runs=10, warmup_runs=2)
    run_performance_test("Numba (parallel, no fastmath)", step_kernel_no_fastmath, num_runs=10, warmup_runs=2)
    run_performance_test("Numba (parallel, fastmath) - original", step_kernel, num_runs=10, warmup_runs=2)

def print_system_info():
    """Print system information for context"""
    print("System Information:")
    print(f"  CPU cores: {mp.cpu_count()}")
    print(f"  Numba version: {numba.__version__}")
    print(f"  NumPy version: {np.__version__}")
    print(f"  Python version: {'.'.join(map(str, (3, 8, 0)))}")  # You might want to get this dynamically
    
    # Check Numba configuration
    print(f"  Numba JIT enabled: {not numba.config.DISABLE_JIT}")
    print(f"  Numba threads: {numba.get_num_threads()}")
    print(f"  Numba parallel: {numba.config.PARALLEL_DIAGNOSTICS}")

if __name__ == "__main__":
    print_system_info()
    test_numba_configurations()
    test_numba_flags()
    
    print("\n" + "="*60)
    print("Performance Test Complete!")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. First run with Numba includes compilation overhead")
    print("2. Subsequent runs use cached compiled code")
    print("3. Threading performance depends on CPU cores and workload")
    print("4. fastmath=True can provide significant speedup for math-heavy code")
    print("5. parallel=True helps with loop-based operations")

# Quick Numba test function
def quick_numba_test():
    """Quick test to see immediate Numba effects."""
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

# Run the test
quick_numba_test() 