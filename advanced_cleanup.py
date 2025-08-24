#!/usr/bin/env python3
"""
Advanced Results Cleanup Script

Can clean up at both the directory level and individual run level:
1. Remove entire result sets below threshold
2. Remove individual failed runs within result sets
3. Remove individual low-scoring runs within result sets
4. Keep best N runs from each result set

Usage:
    python advanced_cleanup.py --help
    python advanced_cleanup.py --dir-threshold 950 --dry-run        # Remove whole dirs below 950
    python advanced_cleanup.py --run-threshold 900 --dry-run        # Remove individual runs below 900  
    python advanced_cleanup.py --keep-best 3 --dry-run              # Keep only best 3 runs per set
    python advanced_cleanup.py --remove-failed-runs --dry-run       # Remove failed individual runs
"""

import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path
import subprocess

def get_size_mb(path):
    """Get directory size in MB."""
    try:
        result = subprocess.run(['du', '-sm', str(path)], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return int(result.stdout.split()[0])
    except:
        pass
    return 0

def analyze_run_directory(run_dir):
    """Analyze a single run directory."""
    info = {
        'path': run_dir,
        'name': run_dir.name,
        'size_mb': get_size_mb(run_dir),
        'best_score': None,
        'status': 'unknown'
    }
    
    aggregated_file = run_dir / 'aggregated_results.pkl'
    if aggregated_file.exists():
        try:
            with open(aggregated_file, 'rb') as f:
                data = pickle.load(f)
            
            if 'summary' in data and 'best_overall_score' in data['summary']:
                info['best_score'] = data['summary']['best_overall_score']
                info['status'] = 'success'
            else:
                info['status'] = 'incomplete'
        except:
            info['status'] = 'corrupted'
    else:
        # Check for process files to distinguish failed vs incomplete
        process_files = list(run_dir.glob('process_*_results.pkl'))
        if process_files:
            info['status'] = 'incomplete'
        else:
            info['status'] = 'failed'
    
    return info

def analyze_result_set(result_dir):
    """Analyze a complete result set (multiple_runs directory)."""
    info = {
        'path': result_dir,
        'name': result_dir.name,
        'type': 'single_run',
        'size_mb': get_size_mb(result_dir),
        'runs': [],
        'best_score': None,
        'status': 'unknown'
    }
    
    if 'multiple_runs' in result_dir.name:
        info['type'] = 'multiple_runs'
        
        # Analyze each run
        run_dirs = sorted([d for d in result_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('run_')])
        
        for run_dir in run_dirs:
            run_info = analyze_run_directory(run_dir)
            info['runs'].append(run_info)
        
        # Get overall stats
        if info['runs']:
            scores = [r['best_score'] for r in info['runs'] if r['best_score'] is not None]
            if scores:
                info['best_score'] = max(scores)
                success_count = sum(1 for r in info['runs'] if r['status'] == 'success')
                if success_count > len(info['runs']) * 0.5:
                    info['status'] = 'success'
                elif success_count > 0:
                    info['status'] = 'partial'
                else:
                    info['status'] = 'failed'
            else:
                info['status'] = 'failed'
        else:
            info['status'] = 'failed'
    else:
        # Single run - analyze directly
        run_info = analyze_run_directory(result_dir)
        info.update(run_info)
        info['runs'] = [run_info]
    
    return info

def main():
    parser = argparse.ArgumentParser(description="Advanced results cleanup")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    
    # Directory-level cleanup
    parser.add_argument("--dir-threshold", type=int, help="Remove entire result sets below this score")
    
    # Run-level cleanup
    parser.add_argument("--run-threshold", type=int, help="Remove individual runs below this score")
    parser.add_argument("--remove-failed-runs", action="store_true", help="Remove failed individual runs")
    parser.add_argument("--keep-best", type=int, help="Keep only N best runs per result set")
    
    # General options
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--force", action="store_true", help="Skip confirmation")
    
    args = parser.parse_args()
    
    results_path = Path(args.results_dir)
    if not results_path.exists():
        print(f"❌ Directory not found: {results_path}")
        sys.exit(1)
    
    print("🔧 ADVANCED RESULTS CLEANUP")
    print("=" * 40)
    
    # Get result sets
    result_dirs = [d for d in results_path.iterdir() 
                   if d.is_dir() and not d.name.startswith('.')]
    
    print(f"Analyzing {len(result_dirs)} result sets...")
    
    # Analyze all result sets
    result_sets = []
    total_mb = 0
    
    for result_dir in result_dirs:
        print(f"  Analyzing {result_dir.name}...", end="")
        result_set = analyze_result_set(result_dir)
        result_sets.append(result_set)
        total_mb += result_set['size_mb']
        
        status_emoji = {
            'success': '✅',
            'partial': '⚠️', 
            'failed': '❌',
            'incomplete': '❓'
        }.get(result_set['status'], '❓')
        
        run_count = len(result_set['runs'])
        score_str = f" ({result_set['best_score']})" if result_set['best_score'] else ""
        print(f" {status_emoji}{score_str} {run_count} runs, {result_set['size_mb']}MB")
    
    print(f"\nTotal space: {total_mb}MB ({total_mb/1024:.1f}GB)")
    
    # Plan cleanup
    dirs_to_delete = []
    runs_to_delete = []
    
    # Directory-level cleanup
    if args.dir_threshold:
        for result_set in result_sets:
            if result_set['best_score'] and result_set['best_score'] < args.dir_threshold:
                dirs_to_delete.append((result_set, f"Score {result_set['best_score']} < {args.dir_threshold}"))
    
    # Run-level cleanup (only for sets not being deleted entirely)
    remaining_sets = [rs for rs in result_sets 
                     if not any(rs['path'] == d[0]['path'] for d, _ in dirs_to_delete)]
    
    for result_set in remaining_sets:
        if result_set['type'] == 'multiple_runs':
            runs = result_set['runs'].copy()
            
            # Remove failed runs
            if args.remove_failed_runs:
                for run in runs:
                    if run['status'] == 'failed':
                        runs_to_delete.append((run, "Failed run (no results)"))
            
            # Remove low-scoring runs
            if args.run_threshold:
                for run in runs:
                    if run['best_score'] and run['best_score'] < args.run_threshold:
                        runs_to_delete.append((run, f"Score {run['best_score']} < {args.run_threshold}"))
            
            # Keep only best N runs
            if args.keep_best:
                successful_runs = [(run, run['best_score'] or 0) for run in runs 
                                  if run['status'] == 'success' and 
                                  not any(run['path'] == r[0]['path'] for r, _ in runs_to_delete)]
                successful_runs.sort(key=lambda x: x[1], reverse=True)
                
                if len(successful_runs) > args.keep_best:
                    for run, score in successful_runs[args.keep_best:]:
                        runs_to_delete.append((run, f"Beyond best {args.keep_best} runs (score {score})"))
    
    # Show cleanup plan
    dir_delete_mb = sum(result_set['size_mb'] for result_set, _ in dirs_to_delete)
    run_delete_mb = sum(run['size_mb'] for run, _ in runs_to_delete)
    total_delete_mb = dir_delete_mb + run_delete_mb
    
    print(f"\n📊 CLEANUP PLAN")
    print("=" * 40)
    print(f"Result sets to delete: {len(dirs_to_delete)} ({dir_delete_mb}MB)")
    print(f"Individual runs to delete: {len(runs_to_delete)} ({run_delete_mb}MB)")
    print(f"Total space to free: {total_delete_mb}MB ({total_delete_mb/1024:.1f}GB)")
    print(f"Remaining space: {(total_mb-total_delete_mb)}MB ({(total_mb-total_delete_mb)/1024:.1f}GB)")
    
    if dirs_to_delete:
        print(f"\n🗑️ RESULT SETS TO DELETE:")
        for result_set, reason in dirs_to_delete:
            print(f"  ❌ {result_set['name']} - {result_set['size_mb']}MB - {reason}")
    
    if runs_to_delete:
        print(f"\n🗑️ INDIVIDUAL RUNS TO DELETE:")
        for run, reason in runs_to_delete:
            print(f"  ❌ {run['name']} - {run['size_mb']}MB - {reason}")
    
    if not dirs_to_delete and not runs_to_delete:
        print("✅ No cleanup needed with current criteria!")
        
        # Show suggestions
        print(f"\n💡 CLEANUP SUGGESTIONS:")
        all_scores = []
        for rs in result_sets:
            if rs['best_score']:
                all_scores.append(rs['best_score'])
        
        if all_scores:
            all_scores.sort()
            low = all_scores[0]
            mid = all_scores[len(all_scores)//2] if len(all_scores) > 1 else low
            print(f"   --dir-threshold {mid} would remove result sets below median score")
            
            # Count runs
            all_run_scores = []
            for rs in result_sets:
                for run in rs['runs']:
                    if run['best_score']:
                        all_run_scores.append(run['best_score'])
            
            if len(all_run_scores) > 10:
                all_run_scores.sort()
                run_low = all_run_scores[len(all_run_scores)//4]  # 25th percentile
                print(f"   --run-threshold {run_low} would remove individual runs below 25th percentile")
                print(f"   --keep-best 3 would keep only 3 best runs per result set")
        
        return
    
    if args.dry_run:
        print(f"\n🔍 DRY RUN - Would free {total_delete_mb}MB ({total_delete_mb/1024:.1f}GB)")
        return
    
    # Confirm
    if not args.force:
        total_items = len(dirs_to_delete) + len(runs_to_delete)
        print(f"\n⚠️ Delete {total_items} items and free {total_delete_mb}MB?")
        if input("Continue? (y/N): ").lower() not in ['y', 'yes']:
            print("Cancelled.")
            return
    
    # Delete
    print(f"\n🗑️ Performing cleanup...")
    deleted = 0
    freed_mb = 0
    
    # Delete entire result sets first
    for result_set, reason in dirs_to_delete:
        try:
            print(f"  Deleting result set {result_set['name']} ({result_set['size_mb']}MB)...", end="")
            shutil.rmtree(result_set['path'])
            deleted += 1
            freed_mb += result_set['size_mb']
            print(" ✅")
        except Exception as e:
            print(f" ❌ Error: {e}")
    
    # Delete individual runs
    for run, reason in runs_to_delete:
        try:
            print(f"  Deleting run {run['name']} ({run['size_mb']}MB)...", end="")
            shutil.rmtree(run['path'])
            deleted += 1
            freed_mb += run['size_mb']
            print(" ✅")
        except Exception as e:
            print(f" ❌ Error: {e}")
    
    print(f"\n🎉 CLEANUP COMPLETE!")
    print(f"✅ Deleted: {deleted} items")
    print(f"💾 Freed: {freed_mb}MB ({freed_mb/1024:.1f}GB)")

if __name__ == "__main__":
    main()