#!/usr/bin/env python3
"""
Quick Results Cleanup Script

Fast cleanup focusing on the most common cleanup needs:
1. Remove failed runs (no aggregated_results.pkl)
2. Remove low-scoring runs (below threshold)
3. Keep only N most recent result sets

Usage:
    python quick_cleanup.py --help
    python quick_cleanup.py --failed-only --dry-run       # Remove only failed runs
    python quick_cleanup.py --threshold 900 --dry-run     # Remove scores < 900
    python quick_cleanup.py --keep-recent 3 --dry-run     # Keep only 3 newest
"""

import argparse
import os
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path
import re

def format_size(bytes_size: int) -> str:
    """Format bytes into human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}PB"

def get_dir_size_fast(path: Path) -> int:
    """Quick directory size calculation."""
    try:
        # Use du command for speed
        import subprocess
        result = subprocess.run(['du', '-sb', str(path)], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return int(result.stdout.split()[0])
    except:
        pass
    
    # Fallback to Python method
    total = 0
    try:
        for file in path.rglob('*'):
            if file.is_file():
                total += file.stat().st_size
    except:
        pass
    return total

def extract_timestamp(dir_name: str) -> datetime:
    """Extract timestamp from directory name."""
    match = re.search(r'(\d{8}_\d{6})', dir_name)
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
        except:
            pass
    return datetime.min

def get_best_score_fast(result_dir: Path) -> tuple:
    """Quickly get best score and status for a result directory."""
    # For multiple_runs directories
    if 'multiple_runs' in result_dir.name:
        best_score = None
        successful_runs = 0
        total_runs = 0
        
        # Check each run subdirectory
        for run_dir in result_dir.glob('run_*'):
            if not run_dir.is_dir():
                continue
                
            total_runs += 1
            aggregated_file = run_dir / 'aggregated_results.pkl'
            
            if aggregated_file.exists():
                try:
                    with open(aggregated_file, 'rb') as f:
                        data = pickle.load(f)
                    score = data.get('summary', {}).get('best_overall_score')
                    if score is not None:
                        successful_runs += 1
                        if best_score is None or score > best_score:
                            best_score = score
                except:
                    continue
        
        if successful_runs == 0:
            return None, 'failed'
        elif successful_runs < total_runs * 0.5:  # Less than 50% success
            return best_score, 'mostly_failed'
        else:
            return best_score, 'success'
    
    # For single run directories
    else:
        aggregated_file = result_dir / 'aggregated_results.pkl'
        if aggregated_file.exists():
            try:
                with open(aggregated_file, 'rb') as f:
                    data = pickle.load(f)
                score = data.get('summary', {}).get('best_overall_score')
                return score, 'success' if score is not None else 'incomplete'
            except:
                return None, 'corrupted'
        else:
            return None, 'failed'

def main():
    parser = argparse.ArgumentParser(description="Quick cleanup of GA results")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--threshold", type=int, help="Delete runs below this score")
    parser.add_argument("--failed-only", action="store_true", help="Only delete failed runs")
    parser.add_argument("--keep-recent", type=int, help="Keep only N most recent")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't delete")
    parser.add_argument("--force", action="store_true", help="Skip confirmation")
    
    args = parser.parse_args()
    
    results_path = Path(args.results_dir)
    if not results_path.exists():
        print(f"❌ Directory not found: {results_path}")
        sys.exit(1)
    
    print("🚀 QUICK RESULTS CLEANUP")
    print("=" * 40)
    
    # Get all result directories
    result_dirs = [d for d in results_path.iterdir() 
                   if d.is_dir() and not d.name.startswith('.')]
    
    print(f"Found {len(result_dirs)} directories")
    
    # Analyze directories
    dir_info = []
    print("Analyzing directories...")
    
    for i, dir_path in enumerate(result_dirs):
        print(f"  {i+1}/{len(result_dirs)}: {dir_path.name}", end="...")
        
        timestamp = extract_timestamp(dir_path.name)
        best_score, status = get_best_score_fast(dir_path)
        size = get_dir_size_fast(dir_path)
        
        dir_info.append({
            'path': dir_path,
            'name': dir_path.name,
            'timestamp': timestamp,
            'best_score': best_score,
            'status': status,
            'size': size
        })
        
        status_emoji = {'success': '✅', 'failed': '❌', 'mostly_failed': '⚠️', 'incomplete': '❓'}.get(status, '❓')
        score_str = f" (Score: {best_score})" if best_score else ""
        print(f" {status_emoji}{score_str} {format_size(size)}")
    
    # Sort by timestamp (newest first)
    dir_info.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Determine what to delete
    to_delete = []
    total_size = sum(d['size'] for d in dir_info)
    
    for i, info in enumerate(dir_info):
        should_delete = False
        reason = ""
        
        # Keep recent filter (applied first)
        if args.keep_recent and i >= args.keep_recent:
            should_delete = True
            reason = f"Beyond {args.keep_recent} most recent"
        
        # Failed only filter
        elif args.failed_only and info['status'] == 'failed':
            should_delete = True
            reason = "Failed run (no valid results)"
        
        # Threshold filter
        elif args.threshold and info['best_score'] is not None and info['best_score'] < args.threshold:
            should_delete = True
            reason = f"Score {info['best_score']} < {args.threshold}"
        
        # General cleanup rules
        elif not args.failed_only and not args.threshold and not args.keep_recent:
            if info['status'] == 'failed':
                should_delete = True
                reason = "Failed run"
            elif info['status'] == 'mostly_failed':
                should_delete = True
                reason = "Mostly failed runs"
        
        if should_delete:
            to_delete.append((info, reason))
    
    if not to_delete:
        print("\n✅ No directories need cleanup!")
        return
    
    # Show summary
    delete_size = sum(info['size'] for info, _ in to_delete)
    keep_count = len(dir_info) - len(to_delete)
    keep_size = total_size - delete_size
    
    print(f"\n📊 CLEANUP SUMMARY")
    print("=" * 40)
    print(f"Total directories: {len(dir_info)}")
    print(f"To delete: {len(to_delete)}")
    print(f"To keep: {keep_count}")
    print(f"Current size: {format_size(total_size)}")
    print(f"Will free: {format_size(delete_size)} ({delete_size/total_size*100:.1f}%)")
    print(f"Remaining: {format_size(keep_size)}")
    
    print(f"\n🗑️ DIRECTORIES TO DELETE:")
    for info, reason in to_delete:
        status_emoji = {'success': '✅', 'failed': '❌', 'mostly_failed': '⚠️', 'incomplete': '❓'}.get(info['status'], '❓')
        score_str = f" (Score: {info['best_score']})" if info['best_score'] else ""
        print(f"  {status_emoji} {info['name']}{score_str} - {format_size(info['size'])} - {reason}")
    
    if args.dry_run:
        print(f"\n🔍 DRY RUN - Would free {format_size(delete_size)}")
        return
    
    # Confirmation
    if not args.force:
        print(f"\n⚠️ Delete {len(to_delete)} directories and free {format_size(delete_size)}?")
        if input("Continue? (y/N): ").lower() not in ['y', 'yes']:
            print("Cancelled.")
            return
    
    # Delete directories
    print(f"\n🗑️ Deleting {len(to_delete)} directories...")
    deleted = 0
    freed = 0
    
    for info, reason in to_delete:
        try:
            print(f"  Deleting {info['name']}...", end="")
            shutil.rmtree(info['path'])
            deleted += 1
            freed += info['size']
            print(" ✅")
        except Exception as e:
            print(f" ❌ Error: {e}")
    
    print(f"\n🎉 CLEANUP COMPLETE!")
    print(f"✅ Deleted: {deleted} directories")
    print(f"💾 Freed: {format_size(freed)}")

if __name__ == "__main__":
    main()