#!/usr/bin/env python3
"""
Script to run adaptive_tmax_fully_optimized.py multiple times with configurable parameters.

Usage:
    # Normal mode - run exactly N times
    python run_multiple_ga.py --runs 5 --config medium --opt-level 3
    python run_multiple_ga.py --runs 10 --config small --opt-level 2 --processes 4 --generations 50
    python run_multiple_ga.py --runs 3 --config large --strategy exponential --results-dir custom_dir
    
    # Continuous mode - keep trying until N successful runs with score >= threshold
    python run_multiple_ga.py --runs 5 --config medium --continuous --threshold 950
    python run_multiple_ga.py --runs 3 --config small --continuous --threshold 360 --max-attempts 20
    
    # Memory management options
    python run_multiple_ga.py --runs 5 --config medium --memory-limit 8000
    python run_multiple_ga.py --runs 10 --config small --no-clear-memory

Options:
    --runs N              Number of GA runs (required)
    --config CONFIG       GA configuration: small, medium, large, etc. (default: medium)
    --opt-level LEVEL     Optimization level 1-4: 1=basic, 2=+early_term, 3=+reduced_prec, 4=all (default: 3)
    --processes N         Number of parallel processes (default: CPU count)
    --generations N       Generations per process (default: from config)
    --strategy STRATEGY   TMAX adaptation: progressive, exponential, sigmoid (default: progressive)
    --results-dir DIR     Base results directory (default: timestamped)
    --continuous          Run until target successful runs achieved
    --threshold SCORE     Min score for continuous mode (default: 900)
    --max-attempts N      Max attempts in continuous mode (default: 50)
    --no-clear-memory     Disable memory clearing between runs
    --memory-limit MB     Force cleanup if memory exceeds limit
"""

import argparse
import time
import subprocess
import sys
import pickle
import numpy as np
import gc
import os
import shutil
from pathlib import Path
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ Warning: psutil not available. Memory monitoring will be limited.")
    print("   Install with: pip install psutil")

def clear_memory_cache(verbose=True):
    """Clear memory cache and force garbage collection."""
    memory_before = None
    memory_after = None
    
    if verbose and PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Force garbage collection
    collected_objects = 0
    for generation in range(3):
        collected_objects += gc.collect()
    
    # Clear numpy cache
    try:
        import numba
        numba.cuda.close()  # Clear CUDA cache if using GPU
    except:
        pass
    
    # Force Python to release memory back to OS (platform specific)
    try:
        import ctypes
        if hasattr(ctypes, 'c_int'):
            libc = ctypes.CDLL("libc.so.6")
            if hasattr(libc, 'malloc_trim'):
                libc.malloc_trim(0)
    except:
        pass
    
    if verbose:
        if PSUTIL_AVAILABLE:
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_freed = memory_before - memory_after
            print(f"  Memory cleared: {memory_before:.1f}MB → {memory_after:.1f}MB "
                  f"({memory_freed:+.1f}MB freed)")
        else:
            print(f"  Memory cleared: {collected_objects} objects freed by garbage collector")
    
    return memory_before, memory_after

def get_memory_usage():
    """Get current memory usage in MB."""
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    else:
        return None

def extract_best_dna_from_run(run_results_dir):
    """Extract the best DNA and related info from a completed GA run with aggressive memory cleanup."""
    run_path = Path(run_results_dir)
    
    try:
        # Look for aggregated_results.pkl first
        aggregated_file = run_path / "aggregated_results.pkl"
        if aggregated_file.exists():
            data = None
            best_dna_record = None
            best_score = -1
            
            try:
                with open(aggregated_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Find the best DNA across all tested individuals
                for dna_record in data.get('all_dna_tested', []):
                    total_score = dna_record.get('total_score', 0)
                    if total_score > best_score:
                        best_score = total_score
                        # Copy only what we need to avoid holding references
                        best_dna_record = {
                            'dna': dna_record['dna'].copy() if hasattr(dna_record['dna'], 'copy') else np.array(dna_record['dna']),
                            'total_score': total_score,
                            'exp_score': dna_record['exp_score'],
                            'cont_score': dna_record['cont_score'],
                            'generation': dna_record['generation'],
                            'process_id': dna_record['process_id'],
                            'individual_id': dna_record['individual_id']
                        }
            finally:
                # Aggressively clean up the large data structure
                if data is not None:
                    del data
                gc.collect()
            
            if best_dna_record:
                result = {
                    'dna_vector': best_dna_record['dna'].tolist() if hasattr(best_dna_record['dna'], 'tolist') else list(best_dna_record['dna']),
                    'total_score': best_dna_record['total_score'],
                    'exp_score': best_dna_record['exp_score'],
                    'cont_score': best_dna_record['cont_score'],
                    'generation': best_dna_record['generation'],
                    'process_id': best_dna_record['process_id'],
                    'individual_id': best_dna_record['individual_id'],
                    'non_zero_weights': int(np.count_nonzero(best_dna_record['dna']))
                }
                # Clean up the numpy array
                del best_dna_record
                gc.collect()
                return result
        
        # If no aggregated results, try to find process files
        process_files = list(run_path.glob("process_*_results.pkl"))
        if process_files:
            best_overall = None
            best_score = -1
            
            for process_file in process_files:
                process_data = None
                try:
                    with open(process_file, 'rb') as f:
                        process_data = pickle.load(f)
                    
                    for dna_record in process_data.get('all_tested_dna', []):
                        total_score = dna_record.get('total_score', 0)
                        if total_score > best_score:
                            best_score = total_score
                            # Copy only what we need
                            best_overall = {
                                'dna': dna_record['dna'].copy() if hasattr(dna_record['dna'], 'copy') else np.array(dna_record['dna']),
                                'total_score': total_score,
                                'exp_score': dna_record['exp_score'],
                                'cont_score': dna_record['cont_score'],
                                'generation': dna_record['generation'],
                                'process_id': dna_record['process_id'],
                                'individual_id': dna_record['individual_id']
                            }
                except:
                    continue
                finally:
                    # Clean up each process file's data
                    if process_data is not None:
                        del process_data
                    gc.collect()
            
            if best_overall:
                result = {
                    'dna_vector': best_overall['dna'].tolist() if hasattr(best_overall['dna'], 'tolist') else list(best_overall['dna']),
                    'total_score': best_overall['total_score'],
                    'exp_score': best_overall['exp_score'],
                    'cont_score': best_overall['cont_score'],
                    'generation': best_overall['generation'],
                    'process_id': best_overall['process_id'],
                    'individual_id': best_overall['individual_id'],
                    'non_zero_weights': int(np.count_nonzero(best_overall['dna']))
                }
                # Clean up
                del best_overall
                gc.collect()
                return result
        
        return {
            'error': 'No valid results found',
            'dna_vector': None,
            'total_score': 0
        }
        
    except Exception as e:
        # Force cleanup on error too
        gc.collect()
        return {
            'error': f'Failed to extract DNA: {str(e)}',
            'dna_vector': None,
            'total_score': 0
        }

def run_ga_multiple_times(num_runs: int, config: str, opt_level: int, 
                         processes: int = None, generations: int = None,
                         strategy: str = "progressive", base_results_dir: str = None,
                         clear_memory: bool = True, memory_limit_mb: int = None):
    """
    Run the GA multiple times with the specified parameters.
    
    Args:
        num_runs: Number of times to run the GA
        config: GA configuration (small, medium, large, etc.)
        opt_level: Optimization level (1-4)
        processes: Number of processes (default: system CPU count)
        generations: Number of generations per process (default: from config)
        strategy: TMAX adaptation strategy
        base_results_dir: Base directory for results (default: timestamped)
        clear_memory: Whether to clear memory between runs (default: True)
        memory_limit_mb: Memory limit in MB; if exceeded, force memory clearing
    """
    
    # Create base results directory if not specified
    if base_results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_results_dir = f"results/multiple_runs_{config}_opt{opt_level}_{timestamp}"
    
    base_path = Path(base_results_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    print("🚀 MULTIPLE GA RUNNER 🚀")
    print("=" * 60)
    print(f"Number of runs: {num_runs}")
    print(f"Configuration: {config}")
    print(f"Optimization level: {opt_level}")
    print(f"Processes: {processes or 'default (CPU count)'}")
    print(f"Generations: {generations or 'default (from config)'}")
    print(f"Strategy: {strategy}")
    print(f"Memory clearing: {'enabled' if clear_memory else 'disabled'}")
    if memory_limit_mb:
        print(f"Memory limit: {memory_limit_mb} MB")
    print(f"Base results directory: {base_path}")
    initial_memory = get_memory_usage()
    if initial_memory is not None:
        print(f"Initial memory usage: {initial_memory:.1f} MB")
    print("=" * 60)
    
    results_summary = []
    total_start_time = time.time()
    
    for run_idx in range(1, num_runs + 1):
        # Check memory before run
        memory_before_run = get_memory_usage()
        
        # Check if memory limit is exceeded
        if memory_limit_mb and memory_before_run is not None and memory_before_run > memory_limit_mb:
            print(f"\n⚠️ Memory limit exceeded ({memory_before_run:.1f}MB > {memory_limit_mb}MB)")
            print("  Forcing memory cleanup...")
            clear_memory_cache(verbose=True)
            memory_before_run = get_memory_usage()
        
        if memory_before_run is not None:
            print(f"\n🏃‍♂️ Starting RUN {run_idx}/{num_runs} (Memory: {memory_before_run:.1f}MB)")
        else:
            print(f"\n🏃‍♂️ Starting RUN {run_idx}/{num_runs}")
        print("-" * 40)
        
        run_start_time = time.time()
        
        # Create results directory for this run
        run_results_dir = base_path / f"run_{run_idx:03d}"
        
        # Build command
        cmd = [
            sys.executable, "adaptive_tmax_fully_optimized.py",
            "--config", config,
            "--opt-level", str(opt_level),
            "--strategy", strategy,
            "--results-dir", str(run_results_dir)
        ]
        
        if processes:
            cmd.extend(["--processes", str(processes)])
        if generations:
            cmd.extend(["--generations", str(generations)])
        if clear_memory:
            cmd.append("--clear-cache")
        
        try:
            # Run the GA
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            run_end_time = time.time()
            run_duration = run_end_time - run_start_time
            
            print(f"✅ RUN {run_idx} completed successfully in {run_duration:.2f} seconds")
            
            # Extract performance info from stdout if available
            stdout_lines = result.stdout.strip().split('\n')
            performance_info = {}
            
            for line in stdout_lines:
                if "Total individuals tested:" in line:
                    performance_info['total_individuals'] = line.split()[-1].replace(',', '')
                elif "Performance:" in line and "individuals/second" in line:
                    performance_info['individuals_per_sec'] = line.split()[1]
                elif "Best overall score:" in line:
                    performance_info['best_score'] = line.split()[-1]
                elif "Time speedup from adaptive TMAX:" in line:
                    performance_info['speedup'] = line.split()[-1].replace('x', '')
            
            # Extract best DNA from this run
            best_dna_info = extract_best_dna_from_run(run_results_dir)
            
            # Memory cleanup after successful run
            memory_after_run = get_memory_usage()
            memory_info = {
                'before_run': memory_before_run,
                'after_run': memory_after_run,
                'memory_increase': (memory_after_run - memory_before_run) if (memory_after_run is not None and memory_before_run is not None) else None
            }
            
            if clear_memory and run_idx < num_runs:  # Don't clear after last run
                if memory_after_run is not None and memory_info['memory_increase'] is not None:
                    print(f"  Memory after run: {memory_after_run:.1f}MB (+{memory_info['memory_increase']:+.1f}MB)")
                print("  Clearing memory cache...")
                clear_memory_cache(verbose=True)
                memory_info['after_cleanup'] = get_memory_usage()
            
            results_summary.append({
                'run': run_idx,
                'duration': run_duration,
                'status': 'success',
                'results_dir': str(run_results_dir),
                'performance': performance_info,
                'best_dna': best_dna_info,
                'memory': memory_info
            })
            
        except subprocess.CalledProcessError as e:
            run_end_time = time.time()
            run_duration = run_end_time - run_start_time
            
            print(f"❌ RUN {run_idx} failed after {run_duration:.2f} seconds")
            print(f"Error: {e}")
            if e.stdout:
                print(f"STDOUT:\n{e.stdout}")
            if e.stderr:
                print(f"STDERR:\n{e.stderr}")
            
            results_summary.append({
                'run': run_idx,
                'duration': run_duration,
                'status': 'failed',
                'error': str(e),
                'results_dir': str(run_results_dir)
            })
        
        except Exception as e:
            run_end_time = time.time()
            run_duration = run_end_time - run_start_time
            
            print(f"💥 RUN {run_idx} crashed after {run_duration:.2f} seconds")
            print(f"Unexpected error: {e}")
            
            results_summary.append({
                'run': run_idx,
                'duration': run_duration,
                'status': 'crashed',
                'error': str(e),
                'results_dir': str(run_results_dir)
            })
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Print final summary
    print("\n" + "=" * 60)
    print("🎉 ALL RUNS COMPLETED 🎉")
    print("=" * 60)
    
    successful_runs = sum(1 for r in results_summary if r['status'] == 'success')
    failed_runs = sum(1 for r in results_summary if r['status'] != 'success')
    
    print(f"Total time: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours)")
    print(f"Successful runs: {successful_runs}/{num_runs}")
    print(f"Failed runs: {failed_runs}/{num_runs}")
    print(f"Average time per run: {total_duration/num_runs:.2f} seconds")
    
    # Performance stats for successful runs
    if successful_runs > 0:
        successful_results = [r for r in results_summary if r['status'] == 'success']
        avg_duration = sum(r['duration'] for r in successful_results) / len(successful_results)
        
        print(f"\nSuccessful run statistics:")
        print(f"  Average duration: {avg_duration:.2f} seconds")
        
        # Try to extract performance metrics and best DNA info
        best_scores = []
        speedups = []
        individuals_per_sec = []
        best_dnas = []
        
        for r in successful_results:
            perf = r.get('performance', {})
            if 'best_score' in perf:
                try:
                    best_scores.append(float(perf['best_score']))
                except:
                    pass
            if 'speedup' in perf:
                try:
                    speedups.append(float(perf['speedup']))
                except:
                    pass
            if 'individuals_per_sec' in perf:
                try:
                    individuals_per_sec.append(float(perf['individuals_per_sec']))
                except:
                    pass
            
            # Collect best DNA info
            best_dna = r.get('best_dna', {})
            if best_dna and 'dna_vector' in best_dna and best_dna['dna_vector']:
                best_dnas.append(best_dna)
        
        if best_scores:
            print(f"  Best scores: min={min(best_scores):.1f}, max={max(best_scores):.1f}, "
                  f"avg={sum(best_scores)/len(best_scores):.1f}")
        if speedups:
            print(f"  Speedups: min={min(speedups):.2f}x, max={max(speedups):.2f}x, "
                  f"avg={sum(speedups)/len(speedups):.2f}x")
        if individuals_per_sec:
            print(f"  Performance: min={min(individuals_per_sec):.1f}, max={max(individuals_per_sec):.1f}, "
                  f"avg={sum(individuals_per_sec)/len(individuals_per_sec):.1f} individuals/sec")
        
        # Best DNA summary
        if best_dnas:
            print(f"\n🧬 Best DNA Summary from {len(best_dnas)} successful runs:")
            for i, dna_info in enumerate(best_dnas):
                run_num = [r['run'] for r in successful_results if r.get('best_dna') == dna_info][0]
                print(f"  Run {run_num}: Score={dna_info['total_score']} "
                      f"(Exp:{dna_info['exp_score']}, Cont:{dna_info['cont_score']}) "
                      f"Gen:{dna_info['generation']}, Non-zero:{dna_info['non_zero_weights']}")
                print(f"    DNA: {dna_info['dna_vector']}")
            
            # Find overall best
            overall_best = max(best_dnas, key=lambda x: x['total_score'])
            overall_best_run = [r['run'] for r in successful_results if r.get('best_dna') == overall_best][0]
            print(f"\n🏆 Overall Best DNA from Run {overall_best_run}:")
            print(f"    Score: {overall_best['total_score']} (Exp:{overall_best['exp_score']}, Cont:{overall_best['cont_score']})")
            print(f"    Generation: {overall_best['generation']}, Non-zero weights: {overall_best['non_zero_weights']}")
            print(f"    DNA Vector: {overall_best['dna_vector']}")
    
    # List failed runs
    if failed_runs > 0:
        print(f"\nFailed runs:")
        for r in results_summary:
            if r['status'] != 'success':
                print(f"  Run {r['run']}: {r['status']} - {r.get('error', 'Unknown error')}")
    
    print(f"\nAll results saved in: {base_path}")
    print("=" * 60)
    
    # Save summary to file
    summary_file = base_path / "runs_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Multiple GA Runs Summary\n")
        f.write(f"========================\n")
        f.write(f"Configuration: {config}\n")
        f.write(f"Optimization level: {opt_level}\n")
        f.write(f"Strategy: {strategy}\n")
        f.write(f"Processes: {processes or 'default'}\n")
        f.write(f"Generations: {generations or 'default'}\n")
        f.write(f"Total runs: {num_runs}\n")
        f.write(f"Successful: {successful_runs}\n")
        f.write(f"Failed: {failed_runs}\n")
        f.write(f"Total time: {total_duration:.2f} seconds\n")
        f.write(f"Average time per run: {total_duration/num_runs:.2f} seconds\n")
        
        # Add best DNA information to file
        successful_results = [r for r in results_summary if r['status'] == 'success']
        best_dnas = [r.get('best_dna') for r in successful_results if r.get('best_dna') and r.get('best_dna', {}).get('dna_vector')]
        
        if best_dnas:
            f.write(f"\nBest DNA from Each Run:\n")
            f.write(f"=======================\n")
            for r in successful_results:
                if r.get('best_dna') and r.get('best_dna', {}).get('dna_vector'):
                    dna_info = r['best_dna']
                    f.write(f"Run {r['run']}: Score={dna_info['total_score']} "
                           f"(Exp:{dna_info['exp_score']}, Cont:{dna_info['cont_score']}) "
                           f"Gen:{dna_info['generation']}, Non-zero:{dna_info['non_zero_weights']}\n")
                    f.write(f"  DNA: {dna_info['dna_vector']}\n\n")
            
            # Overall best
            overall_best = max(best_dnas, key=lambda x: x['total_score'])
            overall_best_run = [r['run'] for r in successful_results if r.get('best_dna') == overall_best][0]
            f.write(f"Overall Best DNA (Run {overall_best_run}):\n")
            f.write(f"Score: {overall_best['total_score']} (Exp:{overall_best['exp_score']}, Cont:{overall_best['cont_score']})\n")
            f.write(f"Generation: {overall_best['generation']}, Non-zero weights: {overall_best['non_zero_weights']}\n")
            f.write(f"DNA Vector: {overall_best['dna_vector']}\n\n")
        
        f.write(f"\nDetailed Results:\n")
        for r in results_summary:
            f.write(f"  Run {r['run']}: {r['status']} ({r['duration']:.2f}s) - {r['results_dir']}\n")
            if r['status'] != 'success':
                f.write(f"    Error: {r.get('error', 'Unknown')}\n")
    
    # Save best DNAs as pickle file for easy loading
    if successful_runs > 0:
        successful_results = [r for r in results_summary if r['status'] == 'success']
        best_dnas = [r.get('best_dna') for r in successful_results if r.get('best_dna') and r.get('best_dna', {}).get('dna_vector')]
        
        if best_dnas:
            best_dnas_file = base_path / "best_dnas_summary.pkl"
            with open(best_dnas_file, 'wb') as f:
                pickle.dump({
                    'config': config,
                    'opt_level': opt_level,
                    'strategy': strategy,
                    'num_runs': num_runs,
                    'successful_runs': successful_runs,
                    'best_dnas': best_dnas,
                    'overall_best': max(best_dnas, key=lambda x: x['total_score']) if best_dnas else None,
                    'timestamp': datetime.now().isoformat()
                }, f)
            print(f"Best DNAs summary saved to: {best_dnas_file}")
    
    print(f"Summary saved to: {summary_file}")
    
    return results_summary


def run_continuous_ga(config: str, opt_level: int, num_successful_runs: int = 5,
                     min_score_threshold: int = 900, processes: int = None, 
                     generations: int = None, strategy: str = "progressive",
                     max_attempts: int = 50, clear_memory: bool = True,
                     memory_limit_mb: int = None, base_results_dir: str = None):
    """
    Run GA continuously until achieving the target number of successful runs.
    Discards runs that don't meet the minimum score threshold.
    
    Args:
        config: GA configuration (small, medium, large, etc.)
        opt_level: Optimization level (1-4)
        num_successful_runs: Target number of successful runs to achieve
        min_score_threshold: Minimum score required to keep a run
        processes: Number of processes (default: system CPU count)
        generations: Number of generations per process (default: from config)
        strategy: TMAX adaptation strategy
        max_attempts: Maximum total attempts before giving up
        clear_memory: Whether to clear memory between runs
        memory_limit_mb: Memory limit in MB
        base_results_dir: Base directory for results (default: timestamped)
    """
    # Create base results directory if not specified
    if base_results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_results_dir = f"results/continuous_runs_{config}_opt{opt_level}_thresh{min_score_threshold}_{timestamp}"
    
    base_path = Path(base_results_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    print("🎯 CONTINUOUS GA RUNNER 🎯")
    print("=" * 60)
    print(f"Target successful runs: {num_successful_runs}")
    print(f"Minimum score threshold: {min_score_threshold}")
    print(f"Configuration: {config}")
    print(f"Optimization level: {opt_level}")
    print(f"Processes: {processes or 'default (CPU count)'}")
    print(f"Generations: {generations or 'default (from config)'}")
    print(f"Strategy: {strategy}")
    print(f"Max attempts: {max_attempts}")
    print(f"Memory clearing: {'enabled' if clear_memory else 'disabled'}")
    if memory_limit_mb:
        print(f"Memory limit: {memory_limit_mb} MB")
    print(f"Base results directory: {base_path}")
    initial_memory = get_memory_usage()
    if initial_memory is not None:
        print(f"Initial memory usage: {initial_memory:.1f} MB")
    print("=" * 60)
    
    successful_runs = []
    discarded_runs = []
    failed_runs = []
    attempt = 0
    total_start_time = time.time()
    
    while len(successful_runs) < num_successful_runs and attempt < max_attempts:
        attempt += 1
        
        # Check memory before run
        memory_before_run = get_memory_usage()
        
        # Check if memory limit is exceeded
        if memory_limit_mb and memory_before_run is not None and memory_before_run > memory_limit_mb:
            print(f"\n⚠️ Memory limit exceeded ({memory_before_run:.1f}MB > {memory_limit_mb}MB)")
            print("  Forcing memory cleanup...")
            clear_memory_cache(verbose=True)
            memory_before_run = get_memory_usage()
        
        if memory_before_run is not None:
            print(f"\n🧬 Attempt {attempt}/{max_attempts} (Memory: {memory_before_run:.1f}MB)")
        else:
            print(f"\n🧬 Attempt {attempt}/{max_attempts}")
        
        print(f"   Target: {num_successful_runs - len(successful_runs)} more runs with score ≥ {min_score_threshold}")
        print("-" * 40)
        
        run_start_time = time.time()
        
        # Create temporary results directory for this attempt
        temp_results_dir = base_path / f"temp_attempt_{attempt:03d}"
        
        # Build command
        cmd = [
            sys.executable, "adaptive_tmax_fully_optimized.py",
            "--config", config,
            "--opt-level", str(opt_level),
            "--strategy", strategy,
            "--results-dir", str(temp_results_dir)
        ]
        
        if processes:
            cmd.extend(["--processes", str(processes)])
        if generations:
            cmd.extend(["--generations", str(generations)])
        if clear_memory:
            cmd.append("--clear-cache")
        
        try:
            # Run the GA
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            run_end_time = time.time()
            run_duration = run_end_time - run_start_time
            
            # Extract performance info from stdout if available
            stdout_lines = result.stdout.strip().split('\n')
            performance_info = {}
            
            for line in stdout_lines:
                if "Best overall score:" in line:
                    try:
                        performance_info['best_score'] = float(line.split()[-1])
                    except:
                        performance_info['best_score'] = 0
                elif "Total individuals tested:" in line:
                    performance_info['total_individuals'] = line.split()[-1].replace(',', '')
                elif "Performance:" in line and "individuals/second" in line:
                    performance_info['individuals_per_sec'] = line.split()[1]
                elif "Time speedup from adaptive TMAX:" in line:
                    performance_info['speedup'] = line.split()[-1].replace('x', '')
            
            best_score = performance_info.get('best_score', 0)
            
            print(f"   📊 Completed in {(run_duration//60):.1f}min | Best score: {best_score}")
            
            if best_score >= min_score_threshold:
                # Successful run - keep it
                success_id = len(successful_runs) + 1
                final_results_dir = base_path / f"successful_run_{success_id:03d}"
                
                # Move temp results to final location
                if temp_results_dir.exists():
                    shutil.move(str(temp_results_dir), str(final_results_dir))
                
                # Extract best DNA from this run (this can use a lot of memory)
                best_dna_info = extract_best_dna_from_run(final_results_dir)
                
                run_info = {
                    'run_id': success_id,
                    'attempt_number': attempt,
                    'config': config,
                    'best_score': best_score,
                    'run_duration': run_duration,
                    'results_dir': str(final_results_dir),
                    'performance': performance_info,
                    'best_dna': best_dna_info,
                    'timestamp': datetime.now().isoformat()
                }
                
                successful_runs.append(run_info)
                
                print(f"   ✅ SUCCESS! Keeping run {success_id} (score: {best_score})")
                print(f"      Progress: {len(successful_runs)}/{num_successful_runs} successful runs")
                
                # Force garbage collection after successful DNA extraction
                gc.collect()
                
            else:
                # Score too low - discard
                discard_info = {
                    'attempt_number': attempt,
                    'config': config,
                    'best_score': best_score,
                    'threshold': min_score_threshold,
                    'run_duration': run_duration,
                    'performance': performance_info,
                    'timestamp': datetime.now().isoformat()
                }
                
                discarded_runs.append(discard_info)
                
                # Clean up temp directory
                if temp_results_dir.exists():
                    shutil.rmtree(temp_results_dir)
                
                print(f"   ❌ DISCARDED: Score {best_score} < {min_score_threshold}")
                print(f"      Progress: {len(successful_runs)}/{num_successful_runs} successful runs")
            
            # Memory cleanup after run
            memory_after_run = get_memory_usage()
            
            if clear_memory and attempt < max_attempts and len(successful_runs) < num_successful_runs:
                if memory_after_run is not None and memory_before_run is not None:
                    memory_increase = memory_after_run - memory_before_run
                    print(f"      Memory: {memory_after_run:.1f}MB (+{memory_increase:+.1f}MB)")
                print("      Clearing memory cache...")
                clear_memory_cache(verbose=False)
                
                # Extra aggressive cleanup for successful runs (they use more memory)
                if best_score >= min_score_threshold:
                    # Force multiple garbage collection passes
                    for _ in range(3):
                        gc.collect()
                    
                    # Give OS a chance to reclaim memory
                    time.sleep(0.1)
            
        except subprocess.CalledProcessError as e:
            run_end_time = time.time()
            run_duration = run_end_time - run_start_time
            
            print(f"   ❌ FAILED after {run_duration:.1f}s: {e}")
            
            error_info = {
                'attempt_number': attempt,
                'config': config,
                'error': str(e),
                'run_duration': run_duration,
                'timestamp': datetime.now().isoformat()
            }
            
            failed_runs.append(error_info)
            
            # Clean up temp directory on failure
            if temp_results_dir.exists():
                shutil.rmtree(temp_results_dir)
        
        except Exception as e:
            run_end_time = time.time()
            run_duration = run_end_time - run_start_time
            
            print(f"   💥 CRASHED after {run_duration:.1f}s: {e}")
            
            error_info = {
                'attempt_number': attempt,
                'config': config,
                'error': f"Unexpected error: {str(e)}",
                'run_duration': run_duration,
                'timestamp': datetime.now().isoformat()
            }
            
            failed_runs.append(error_info)
            
            # Clean up temp directory on crash
            if temp_results_dir.exists():
                shutil.rmtree(temp_results_dir)
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Final status
    if len(successful_runs) >= num_successful_runs:
        print(f"\n🎉 TARGET ACHIEVED! {len(successful_runs)} successful runs in {attempt} attempts")
    else:
        print(f"\n⚠️ STOPPED: Max attempts ({max_attempts}) reached with {len(successful_runs)} successful runs")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("🏁 CONTINUOUS RUN COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_duration:.1f} seconds ({total_duration/3600:.1f} hours)")
    print(f"Successful runs: {len(successful_runs)}/{num_successful_runs}")
    print(f"Total attempts: {attempt}")
    print(f"Success rate: {len(successful_runs)/attempt*100:.1f}%")
    print(f"Discarded runs: {len(discarded_runs)} ({len(discarded_runs)/attempt*100:.1f}%)")
    print(f"Failed runs: {len(failed_runs)} ({len(failed_runs)/attempt*100:.1f}%)")
    
    if successful_runs:
        scores = [run['best_score'] for run in successful_runs]
        print(f"\nSuccessful run statistics:")
        print(f"  Best score: {max(scores):.1f}")
        print(f"  Average score: {sum(scores)/len(scores):.1f}")
        print(f"  Score range: {min(scores):.1f} - {max(scores):.1f}")
    
    if discarded_runs:
        discarded_scores = [run['best_score'] for run in discarded_runs]
        print(f"\nDiscarded run statistics:")
        print(f"  Best discarded: {max(discarded_scores):.1f}")
        print(f"  Average discarded: {sum(discarded_scores)/len(discarded_scores):.1f}")
    
    # Save summary
    summary = {
        'experiment_info': {
            'mode': 'continuous',
            'config': config,
            'opt_level': opt_level,
            'target_successful_runs': num_successful_runs,
            'min_score_threshold': min_score_threshold,
            'max_attempts': max_attempts,
            'total_attempts': attempt,
            'total_duration_seconds': total_duration,
            'base_results_dir': str(base_path),
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        },
        'successful_runs': successful_runs,
        'discarded_runs': discarded_runs,
        'failed_runs': failed_runs,
        'summary_stats': {
            'successful': len(successful_runs),
            'discarded': len(discarded_runs),
            'failed': len(failed_runs),
            'total_attempts': attempt,
            'success_rate': len(successful_runs) / attempt * 100 if attempt > 0 else 0,
            'discard_rate': len(discarded_runs) / attempt * 100 if attempt > 0 else 0
        }
    }
    
    if successful_runs:
        scores = [run['best_score'] for run in successful_runs]
        summary['summary_stats'].update({
            'best_score_achieved': max(scores),
            'worst_successful_score': min(scores),
            'average_successful_score': sum(scores) / len(scores)
        })
    
    # Save summary to JSON
    summary_path = base_path / "continuous_experiment_summary.json"
    with open(summary_path, 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    
    # Save text summary
    summary_file = base_path / "continuous_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Continuous GA Run Summary\n")
        f.write(f"=========================\n")
        f.write(f"Configuration: {config}\n")
        f.write(f"Optimization level: {opt_level}\n")
        f.write(f"Min score threshold: {min_score_threshold}\n")
        f.write(f"Target successful runs: {num_successful_runs}\n")
        f.write(f"Total attempts: {attempt}\n")
        f.write(f"Successful: {len(successful_runs)}\n")
        f.write(f"Discarded: {len(discarded_runs)}\n")
        f.write(f"Failed: {len(failed_runs)}\n")
        f.write(f"Success rate: {len(successful_runs)/attempt*100:.1f}%\n")
        f.write(f"Total time: {total_duration:.1f} seconds\n")
        
        if successful_runs:
            f.write(f"\nSuccessful Runs:\n")
            for run in successful_runs:
                f.write(f"  Run {run['run_id']}: Score={run['best_score']:.1f} (Attempt {run['attempt_number']})\n")
                if run.get('best_dna') and run['best_dna'].get('total_score'):
                    dna_info = run['best_dna']
                    f.write(f"    DNA Score: {dna_info['total_score']} (Exp:{dna_info['exp_score']}, Cont:{dna_info['cont_score']})\n")
                    f.write(f"    Non-zero weights: {dna_info['non_zero_weights']}\n")
    
    print(f"\n📁 Results saved to: {base_path}")
    print(f"📊 Summary saved to: {summary_path}")
    print("=" * 60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run adaptive_tmax_fully_optimized.py multiple times",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal mode - run exactly 5 times
  python run_multiple_ga.py --runs 5 --config medium --opt-level 3
  python run_multiple_ga.py --runs 10 --config small --opt-level 2 --processes 4
  
  # Continuous mode - keep trying until 5 successful runs with score ≥ 950
  python run_multiple_ga.py --runs 5 --config medium --continuous --threshold 950
  python run_multiple_ga.py --runs 3 --config small --continuous --threshold 360 --max-attempts 20
  
  # Memory management
  python run_multiple_ga.py --runs 5 --config medium --memory-limit 8000
  python run_multiple_ga.py --runs 10 --config small --no-clear-memory
        """
    )
    
    parser.add_argument("--runs", type=int, required=True,
                       help="Number of times to run the GA")
    parser.add_argument("--config", type=str, default="medium",
                       help="GA configuration (small, medium, large, etc.)")
    parser.add_argument("--opt-level", type=int, choices=[1,2,3,4], default=3,
                       help="Optimization level: 1=basic, 2=+early_term, 3=+reduced_prec, 4=all")
    parser.add_argument("--processes", type=int, default=None,
                       help="Number of processes (default: CPU count)")
    parser.add_argument("--generations", type=int, default=None,
                       help="Number of generations per process (default: from config)")
    parser.add_argument("--strategy", choices=["progressive", "exponential", "sigmoid"], 
                       default="progressive", help="TMAX adaptation strategy")
    parser.add_argument("--results-dir", type=str, default=None,
                       help="Base results directory (default: timestamped dir)")
    parser.add_argument("--no-clear-memory", action="store_true",
                       help="Disable memory clearing between runs")
    parser.add_argument("--memory-limit", type=int, default=None, metavar="MB",
                       help="Memory limit in MB; force cleanup if exceeded")
    
    # Continuous mode options
    parser.add_argument("--continuous", action="store_true",
                       help="Run continuously until target number of successful runs")
    parser.add_argument("--threshold", "-t", type=int, default=900,
                       help="Minimum score threshold for continuous mode (default: 900)")
    parser.add_argument("--max-attempts", type=int, default=50,
                       help="Maximum attempts in continuous mode (default: 50)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.runs <= 0:
        print("Error: Number of runs must be positive")
        sys.exit(1)
    
    if args.continuous:
        # Run continuous mode
        results = run_continuous_ga(
            config=args.config,
            opt_level=args.opt_level,
            num_successful_runs=args.runs,
            min_score_threshold=args.threshold,
            processes=args.processes,
            generations=args.generations,
            strategy=args.strategy,
            max_attempts=args.max_attempts,
            clear_memory=not args.no_clear_memory,
            memory_limit_mb=args.memory_limit,
            base_results_dir=args.results_dir
        )
        
        # Exit based on whether we achieved the target
        successful_count = len(results.get('successful_runs', []))
        if successful_count >= args.runs:
            print(f"\n🎉 Successfully achieved {successful_count} runs above threshold {args.threshold}!")
            sys.exit(0)
        else:
            print(f"\n⚠️  Only achieved {successful_count} out of {args.runs} target runs")
            sys.exit(1)
    else:
        # Run normal mode
        results = run_ga_multiple_times(
            num_runs=args.runs,
            config=args.config,
            opt_level=args.opt_level,
            processes=args.processes,
            generations=args.generations,
            strategy=args.strategy,
            base_results_dir=args.results_dir,
            clear_memory=not args.no_clear_memory,
            memory_limit_mb=args.memory_limit
        )
        
        # Exit with error code if any runs failed
        failed_count = sum(1 for r in results if r['status'] != 'success')
        if failed_count > 0:
            print(f"\n⚠️  {failed_count} out of {args.runs} runs failed")
            sys.exit(1)
        else:
            print(f"\n✅ All {args.runs} runs completed successfully!")
            sys.exit(0)


if __name__ == "__main__":
    main()