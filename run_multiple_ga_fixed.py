#!/usr/bin/env python3
"""
Quick fix for the concurrent GA runner with proper parameter passing.
This addresses the bugs found in the optimized version.
"""

import argparse
import time
import subprocess
import sys
import concurrent.futures
from pathlib import Path
from datetime import datetime

def run_single_ga_job(job_config):
    """Run a single GA job."""
    start_time = time.time()
    
    cmd = [
        sys.executable, "adaptive_tmax_fully_optimized.py",
        "--config", job_config['config'],
        "--opt-level", str(job_config['opt_level']),
        "--strategy", job_config['strategy'],
        "--results-dir", str(job_config['results_dir'])
    ]
    
    if job_config.get('processes'):
        cmd.extend(["--processes", str(job_config['processes'])])
    if job_config.get('generations'):
        cmd.extend(["--generations", str(job_config['generations'])])
    if job_config.get('clear_memory'):
        cmd.append("--clear-cache")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = time.time() - start_time
        
        # Extract best score
        best_score = 0
        for line in result.stdout.strip().split('\n'):
            if "Best overall score:" in line:
                try:
                    best_score = float(line.split()[-1])
                    break
                except:
                    pass
        
        return {
            'job_id': job_config['job_id'],
            'status': 'success',
            'duration': duration,
            'best_score': best_score,
            'results_dir': str(job_config['results_dir'])
        }
        
    except Exception as e:
        duration = time.time() - start_time
        return {
            'job_id': job_config['job_id'],
            'status': 'failed',
            'duration': duration,
            'error': str(e),
            'results_dir': str(job_config['results_dir'])
        }

def main():
    parser = argparse.ArgumentParser(description="Fixed concurrent GA runner")
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--config", type=str, default="medium")
    parser.add_argument("--opt-level", type=int, default=3)
    parser.add_argument("--concurrent", "-c", type=int, default=2)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--strategy", type=str, default="progressive")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--no-clear-memory", action="store_true")
    
    args = parser.parse_args()
    
    # Debug: Show what we're actually using
    print(f"🐛 DEBUG: config='{args.config}', opt_level={args.opt_level}, concurrent={args.concurrent}")
    
    # Create results directory
    if args.results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.results_dir = f"results/fixed_concurrent_{args.config}_opt{args.opt_level}_{timestamp}"
    
    base_path = Path(args.results_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    print("🔧 FIXED CONCURRENT GA RUNNER")
    print("=" * 50)
    print(f"Total runs: {args.runs}")
    print(f"Max concurrent: {args.concurrent}")
    print(f"Configuration: {args.config}")
    print(f"Optimization level: {args.opt_level}")
    print(f"Results directory: {base_path}")
    print("=" * 50)
    
    # Create job configs
    job_configs = []
    for run_idx in range(1, args.runs + 1):
        job_config = {
            'job_id': run_idx,
            'config': args.config,
            'opt_level': args.opt_level,
            'strategy': args.strategy,
            'results_dir': base_path / f"run_{run_idx:03d}",
            'processes': args.processes,
            'generations': args.generations,
            'clear_memory': not args.no_clear_memory
        }
        job_configs.append(job_config)
    
    # Run jobs concurrently
    total_start_time = time.time()
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrent) as executor:
        future_to_job = {executor.submit(run_single_ga_job, job_config): job_config 
                        for job_config in job_configs}
        
        print(f"\n🏃 Running {args.runs} jobs with {args.concurrent} concurrent workers...")
        
        for future in concurrent.futures.as_completed(future_to_job):
            result = future.result()
            results.append(result)
            
            job_id = result['job_id']
            duration = result['duration']
            status = result['status']
            
            if status == 'success':
                best_score = result.get('best_score', 0)
                print(f"✅ Job {job_id:2d}/{args.runs} completed ({duration:6.1f}s) | Score: {best_score}")
            else:
                print(f"❌ Job {job_id:2d}/{args.runs} {status:8s} ({duration:6.1f}s)")
    
    total_duration = time.time() - total_start_time
    successful_jobs = [r for r in results if r['status'] == 'success']
    
    print(f"\n🎉 COMPLETE: {len(successful_jobs)}/{args.runs} successful in {total_duration:.1f}s")
    
    if successful_jobs:
        scores = [r.get('best_score', 0) for r in successful_jobs]
        print(f"Scores: Best={max(scores):.1f}, Avg={sum(scores)/len(scores):.1f}, Range={min(scores):.1f}-{max(scores):.1f}")
        
        # Estimate speedup
        total_job_time = sum(r['duration'] for r in results)
        speedup = total_job_time / total_duration if total_duration > 0 else 1
        print(f"Estimated speedup: {speedup:.1f}x")

if __name__ == "__main__":
    main()