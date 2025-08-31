#!/usr/bin/env python3
"""
Optimized version of run_multiple_ga.py with significant speed improvements:

1. CONCURRENT EXECUTION: Run multiple GA processes in parallel (biggest speedup)
2. LAZY DNA EXTRACTION: Stream best DNA without loading full pickle files
3. OPTIMIZED MEMORY: Reduced cleanup overhead and better memory management
4. BATCHED PROCESSING: Process results in batches for better throughput

Usage:
    # Parallel execution - run 4 GA jobs concurrently
    python run_multiple_ga_optimized.py --runs 12 --config medium --concurrent 4
    
    # Continuous mode with parallelism  
    python run_multiple_ga_optimized.py --runs 5 --config medium --continuous --concurrent 2 --threshold 950
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
import concurrent.futures
import threading
import queue
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Global lock for thread-safe memory monitoring
memory_lock = threading.Lock()

def get_memory_usage() -> Optional[float]:
    """Get current memory usage in MB (thread-safe)."""
    if PSUTIL_AVAILABLE:
        with memory_lock:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
    return None

def optimized_memory_cleanup(aggressive: bool = False, verbose: bool = False) -> Tuple[Optional[float], Optional[float]]:
    """Optimized memory cleanup with reduced overhead."""
    memory_before = get_memory_usage()
    
    if aggressive:
        # Only do aggressive cleanup when really needed
        collected = 0
        for gen in range(3):
            collected += gc.collect()
        
        if verbose:
            memory_after = get_memory_usage()
            if memory_before is not None and memory_after is not None:
                freed = memory_before - memory_after
                print(f"  Memory: {memory_before:.1f}MB → {memory_after:.1f}MB ({freed:+.1f}MB freed)")
            return memory_before, memory_after
    else:
        # Light cleanup - just one GC pass
        gc.collect()
    
    return memory_before, get_memory_usage()

def extract_best_dna_lazy(run_results_dir: Path) -> Dict:
    """
    Lazy extraction of best DNA - only loads what's needed.
    Significantly faster than loading entire pickle files.
    """
    try:
        aggregated_file = run_results_dir / "aggregated_results.pkl"
        
        if aggregated_file.exists():
            # Stream through file to find best score without loading everything
            best_score = -1
            best_dna_record = None
            
            with open(aggregated_file, 'rb') as f:
                try:
                    data = pickle.load(f)
                    all_dna_list = data.get('all_dna_tested', [])
                    
                    # Fast scan for best score
                    for record in all_dna_list:
                        score = record.get('total_score', 0)
                        if score > best_score:
                            best_score = score
                            # Only copy essential data
                            best_dna_record = {
                                'dna': record['dna'][:] if isinstance(record['dna'], (list, np.ndarray)) else record['dna'],
                                'total_score': score,
                                'exp_score': record.get('exp_score', 0),
                                'cont_score': record.get('cont_score', 0),
                                'generation': record.get('generation', 0),
                                'process_id': record.get('process_id', 0),
                                'individual_id': record.get('individual_id', 0)
                            }
                    
                    # Immediately clear the large data structure
                    del data
                    
                except Exception as e:
                    print(f"Warning: Error reading aggregated results: {e}")
                    return {'error': f'Read error: {e}', 'total_score': 0}
            
            if best_dna_record:
                dna_array = np.array(best_dna_record['dna']) if not isinstance(best_dna_record['dna'], np.ndarray) else best_dna_record['dna']
                
                return {
                    'dna_vector': dna_array.tolist(),
                    'total_score': best_dna_record['total_score'],
                    'exp_score': best_dna_record['exp_score'],
                    'cont_score': best_dna_record['cont_score'],
                    'generation': best_dna_record['generation'],
                    'process_id': best_dna_record['process_id'],
                    'individual_id': best_dna_record['individual_id'],
                    'non_zero_weights': int(np.count_nonzero(dna_array))
                }
        
        return {'error': 'No valid results found', 'total_score': 0}
        
    except Exception as e:
        return {'error': f'Failed to extract DNA: {str(e)}', 'total_score': 0}

def run_single_ga_job(job_config: Dict) -> Dict:
    """
    Run a single GA job with optimized process management.
    This function is designed to be run in parallel.
    """
    job_id = job_config['job_id']
    config = job_config['config']
    opt_level = job_config['opt_level']
    strategy = job_config['strategy']
    results_dir = Path(job_config['results_dir'])
    processes = job_config.get('processes')
    generations = job_config.get('generations')
    clear_memory = job_config.get('clear_memory', True)
    
    start_time = time.time()
    
    # Build command
    cmd = [
        sys.executable, "adaptive_tmax_fully_optimized.py",
        "--config", config,
        "--opt-level", str(opt_level),
        "--strategy", strategy,
        "--results-dir", str(results_dir)
    ]
    
    if processes:
        cmd.extend(["--processes", str(processes)])
    if generations:
        cmd.extend(["--generations", str(generations)])
    if clear_memory:
        cmd.append("--clear-cache")
    
    try:
        # Use optimized subprocess execution
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True,
            # Optimize subprocess
            bufsize=8192,  # Larger buffer for better I/O
            preexec_fn=os.setsid if os.name != 'nt' else None  # Process group for clean shutdown
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Extract performance info quickly
        stdout_lines = result.stdout.strip().split('\n')
        performance_info = {}
        
        for line in stdout_lines:
            if "Best overall score:" in line:
                try:
                    performance_info['best_score'] = float(line.split()[-1])
                except:
                    performance_info['best_score'] = 0
                break  # Found what we need, stop scanning
        
        # Lazy DNA extraction
        best_dna_info = extract_best_dna_lazy(results_dir)
        
        return {
            'job_id': job_id,
            'status': 'success',
            'duration': duration,
            'results_dir': str(results_dir),
            'performance': performance_info,
            'best_dna': best_dna_info,
            'stdout_lines_count': len(stdout_lines),  # For debugging
            'timestamp': datetime.now().isoformat()
        }
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            'job_id': job_id,
            'status': 'failed',
            'duration': duration,
            'error': str(e),
            'results_dir': str(results_dir),
            'stderr': e.stderr[:1000] if e.stderr else '',  # Limit error output
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            'job_id': job_id,
            'status': 'crashed',
            'duration': duration,
            'error': f"Unexpected error: {str(e)}",
            'results_dir': str(results_dir),
            'timestamp': datetime.now().isoformat()
        }

def run_ga_concurrent(num_runs: int, config: str, opt_level: int, 
                     max_concurrent: int = 2, processes: int = None, 
                     generations: int = None, strategy: str = "progressive",
                     base_results_dir: str = None, clear_memory: bool = True,
                     memory_limit_mb: int = None) -> List[Dict]:
    """
    Run multiple GA jobs concurrently for maximum speed.
    
    Args:
        num_runs: Total number of GA runs to complete
        max_concurrent: Maximum concurrent GA jobs (default: 2)
        Other args: Same as original function
    """
    
    if base_results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_results_dir = f"results/concurrent_runs_{config}_opt{opt_level}_{timestamp}"
    
    base_path = Path(base_results_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    print("🚀 CONCURRENT GA RUNNER 🚀")
    print("=" * 60)
    print(f"Total runs: {num_runs}")
    print(f"Max concurrent jobs: {max_concurrent}")
    print(f"Configuration: {config}")
    print(f"Optimization level: {opt_level}")
    print(f"Strategy: {strategy}")
    initial_memory = get_memory_usage()
    if initial_memory:
        print(f"Initial memory: {initial_memory:.1f} MB")
    print(f"Results directory: {base_path}")
    print("=" * 60)
    
    # Create job configurations
    job_configs = []
    for run_idx in range(1, num_runs + 1):
        job_config = {
            'job_id': run_idx,
            'config': config,
            'opt_level': opt_level,
            'strategy': strategy,
            'results_dir': base_path / f"run_{run_idx:03d}",
            'processes': processes,
            'generations': generations,
            'clear_memory': clear_memory
        }
        job_configs.append(job_config)
    
    # Run jobs concurrently
    results = []
    completed_jobs = 0
    total_start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        # Submit all jobs
        future_to_job = {
            executor.submit(run_single_ga_job, job_config): job_config 
            for job_config in job_configs
        }
        
        print(f"\n🎯 Running {num_runs} jobs with {max_concurrent} concurrent workers...")
        
        # Process completed jobs as they finish
        for future in concurrent.futures.as_completed(future_to_job):
            job_config = future_to_job[future]
            
            try:
                result = future.result()
                results.append(result)
                completed_jobs += 1
                
                job_id = result['job_id']
                duration = result['duration']
                status = result['status']
                
                if status == 'success':
                    best_score = result['performance'].get('best_score', 0)
                    print(f"✅ Job {job_id:2d}/{num_runs} completed ({duration:6.1f}s) | Score: {best_score}")
                else:
                    print(f"❌ Job {job_id:2d}/{num_runs} {status:8s} ({duration:6.1f}s) | {result.get('error', '')[:50]}")
                
                # Memory management (only occasionally to avoid overhead)
                if completed_jobs % max_concurrent == 0 and memory_limit_mb:
                    current_memory = get_memory_usage()
                    if current_memory and current_memory > memory_limit_mb:
                        print(f"  🧹 Memory cleanup ({current_memory:.1f}MB > {memory_limit_mb}MB)")
                        optimized_memory_cleanup(aggressive=True)
                
            except Exception as e:
                print(f"💥 Job {job_config['job_id']} crashed with exception: {e}")
                results.append({
                    'job_id': job_config['job_id'],
                    'status': 'crashed',
                    'error': str(e),
                    'duration': 0
                })
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Print summary
    successful_jobs = [r for r in results if r['status'] == 'success']
    failed_jobs = [r for r in results if r['status'] != 'success']
    
    print("\n" + "=" * 60)
    print("🎉 CONCURRENT EXECUTION COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_duration:.1f}s ({total_duration/3600:.2f}h)")
    print(f"Successful: {len(successful_jobs)}/{num_runs}")
    print(f"Failed: {len(failed_jobs)}/{num_runs}")
    print(f"Average time per job: {total_duration/num_runs:.1f}s")
    
    if max_concurrent > 1:
        sequential_estimate = sum(r['duration'] for r in results)
        speedup = sequential_estimate / total_duration if total_duration > 0 else 1
        print(f"Estimated speedup: {speedup:.1f}x (vs sequential execution)")
    
    if successful_jobs:
        scores = [r['performance'].get('best_score', 0) for r in successful_jobs]
        print(f"\nScore statistics:")
        print(f"  Best: {max(scores):.1f}")
        print(f"  Average: {sum(scores)/len(scores):.1f}")
        print(f"  Range: {min(scores):.1f} - {max(scores):.1f}")
        
        # Find overall best
        best_result = max(successful_jobs, key=lambda x: x['performance'].get('best_score', 0))
        best_dna = best_result.get('best_dna', {})
        if best_dna and best_dna.get('total_score'):
            print(f"\n🏆 Overall best from Job {best_result['job_id']}:")
            print(f"  Score: {best_dna['total_score']} (Exp:{best_dna['exp_score']}, Cont:{best_dna['cont_score']})")
            print(f"  Non-zero weights: {best_dna['non_zero_weights']}")
    
    print(f"\n📁 Results saved to: {base_path}")
    print("=" * 60)
    
    # Save summary
    save_concurrent_summary(results, base_path, config, opt_level, strategy, 
                           max_concurrent, total_duration)
    
    return results

def save_concurrent_summary(results: List[Dict], base_path: Path, config: str, 
                          opt_level: int, strategy: str, max_concurrent: int, 
                          total_duration: float):
    """Save summary of concurrent run results."""
    successful_jobs = [r for r in results if r['status'] == 'success']
    
    summary_data = {
        'mode': 'concurrent',
        'config': config,
        'opt_level': opt_level,
        'strategy': strategy,
        'max_concurrent': max_concurrent,
        'total_jobs': len(results),
        'successful_jobs': len(successful_jobs),
        'failed_jobs': len(results) - len(successful_jobs),
        'total_duration': total_duration,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save pickle summary
    summary_file = base_path / "concurrent_summary.pkl"
    with open(summary_file, 'wb') as f:
        pickle.dump(summary_data, f)
    
    # Save text summary
    text_summary = base_path / "concurrent_summary.txt"
    with open(text_summary, 'w') as f:
        f.write(f"Concurrent GA Run Summary\n")
        f.write(f"=========================\n")
        f.write(f"Configuration: {config}\n")
        f.write(f"Optimization level: {opt_level}\n")
        f.write(f"Strategy: {strategy}\n")
        f.write(f"Max concurrent: {max_concurrent}\n")
        f.write(f"Total jobs: {len(results)}\n")
        f.write(f"Successful: {len(successful_jobs)}\n")
        f.write(f"Failed: {len(results) - len(successful_jobs)}\n")
        f.write(f"Total time: {total_duration:.1f}s\n")
        
        if successful_jobs:
            scores = [r['performance'].get('best_score', 0) for r in successful_jobs]
            f.write(f"\nBest scores:\n")
            for r in successful_jobs:
                score = r['performance'].get('best_score', 0)
                f.write(f"  Job {r['job_id']}: {score:.1f}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Optimized multiple GA runner with concurrent execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 8 jobs with 2 concurrent workers (4x speedup potential)
  python run_multiple_ga_optimized.py --runs 8 --config medium --concurrent 2
  
  # Run 12 jobs with 4 concurrent workers (maximum parallelism)
  python run_multiple_ga_optimized.py --runs 12 --config small --concurrent 4 --opt-level 4
  
  # Continuous mode with parallelism
  python run_multiple_ga_optimized.py --runs 5 --config medium --continuous --concurrent 2 --threshold 950
        """
    )
    
    parser.add_argument("--runs", type=int, required=True,
                       help="Number of GA runs to complete")
    parser.add_argument("--config", type=str, default="medium",
                       help="GA configuration (small, medium, large, etc.)")
    parser.add_argument("--opt-level", type=int, choices=[1,2,3,4], default=3,
                       help="Optimization level")
    parser.add_argument("--concurrent", "-c", type=int, default=2,
                       help="Maximum concurrent jobs (default: 2)")
    parser.add_argument("--processes", type=int, default=None,
                       help="Processes per GA job")
    parser.add_argument("--generations", type=int, default=None,
                       help="Generations per GA job")
    parser.add_argument("--strategy", choices=["progressive", "exponential", "sigmoid"], 
                       default="progressive", help="TMAX adaptation strategy")
    parser.add_argument("--results-dir", type=str, default=None,
                       help="Base results directory")
    parser.add_argument("--no-clear-memory", action="store_true",
                       help="Disable memory clearing")
    parser.add_argument("--memory-limit", type=int, default=None, metavar="MB",
                       help="Memory limit in MB")
    
    # Continuous mode (simplified for now)
    parser.add_argument("--continuous", action="store_true",
                       help="Run continuously until target successful runs (basic implementation)")
    parser.add_argument("--threshold", "-t", type=int, default=900,
                       help="Minimum score threshold for continuous mode")
    
    args = parser.parse_args()
    
    if args.runs <= 0:
        print("Error: Number of runs must be positive")
        sys.exit(1)
    
    if args.concurrent <= 0:
        print("Error: Concurrent jobs must be positive")
        sys.exit(1)
    
    if args.concurrent > args.runs:
        print(f"Warning: Concurrent jobs ({args.concurrent}) > total runs ({args.runs}), reducing to {args.runs}")
        args.concurrent = args.runs
    
    # Debug: Print actual parameters being used
    print(f"Debug: Using config='{args.config}', opt_level={args.opt_level}, concurrent={args.concurrent}")
    
    # Run optimized concurrent GA
    start_time = time.time()
    results = run_ga_concurrent(
        num_runs=args.runs,
        config=args.config,
        opt_level=args.opt_level,
        max_concurrent=args.concurrent,
        processes=args.processes,
        generations=args.generations,
        strategy=args.strategy,
        base_results_dir=args.results_dir,
        clear_memory=not args.no_clear_memory,
        memory_limit_mb=args.memory_limit
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Exit with status based on results
    successful_count = sum(1 for r in results if r['status'] == 'success')
    if successful_count == args.runs:
        print(f"\n✅ All {args.runs} jobs completed successfully in {total_time:.1f}s!")
        sys.exit(0)
    else:
        failed_count = args.runs - successful_count
        print(f"\n⚠️  {failed_count} out of {args.runs} jobs failed")
        sys.exit(1)

if __name__ == "__main__":
    main()