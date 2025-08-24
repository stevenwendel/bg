#!/usr/bin/env python3
"""
Results Directory Cleanup Script

This script cleans up the results directory by removing:
1. Failed runs (no aggregated_results.pkl or errors)
2. Low-performing runs (best score below threshold)
3. Incomplete runs (missing critical files)
4. Old runs (keeping only the N most recent)

Usage:
    python cleanup_results.py --threshold 900 --dry-run  # Preview what will be deleted
    python cleanup_results.py --threshold 900            # Actually delete
    python cleanup_results.py --keep-recent 5            # Keep only 5 most recent result sets
    python cleanup_results.py --failed-only              # Remove only completely failed runs
"""

import argparse
import os
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

def get_directory_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    continue
    except (OSError, PermissionError):
        return 0
    return total

def format_size(bytes_size: int) -> str:
    """Format bytes into human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}PB"

def extract_timestamp_from_path(path: Path) -> Optional[datetime]:
    """Extract timestamp from result directory name."""
    # Pattern: multiple_runs_H_opt4_20250815_010556
    match = re.search(r'(\d{8}_\d{6})', path.name)
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
        except ValueError:
            pass
    return None

def analyze_result_directory(results_dir: Path) -> Dict:
    """Analyze a results directory and extract key information."""
    info = {
        'path': results_dir,
        'name': results_dir.name,
        'size_bytes': get_directory_size(results_dir),
        'timestamp': extract_timestamp_from_path(results_dir),
        'runs': [],
        'status': 'unknown',
        'best_score': None,
        'total_runs': 0,
        'successful_runs': 0,
        'failed_runs': 0,
        'incomplete_runs': 0
    }
    
    # Check if this is a multiple runs directory or single run
    if 'multiple_runs' in results_dir.name:
        # Multiple runs directory
        info['type'] = 'multiple_runs'
        run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
        info['total_runs'] = len(run_dirs)
        
        for run_dir in sorted(run_dirs):
            run_info = analyze_single_run(run_dir)
            info['runs'].append(run_info)
            
            if run_info['status'] == 'success':
                info['successful_runs'] += 1
            elif run_info['status'] == 'failed':
                info['failed_runs'] += 1
            else:
                info['incomplete_runs'] += 1
                
        # Get best score across all runs
        successful_scores = [r['best_score'] for r in info['runs'] if r['best_score'] is not None]
        if successful_scores:
            info['best_score'] = max(successful_scores)
            info['status'] = 'success' if info['successful_runs'] > 0 else 'failed'
        else:
            info['status'] = 'failed'
            
    else:
        # Single run directory
        info['type'] = 'single_run'
        run_info = analyze_single_run(results_dir)
        info.update(run_info)
        info['total_runs'] = 1
        if run_info['status'] == 'success':
            info['successful_runs'] = 1
        else:
            info['failed_runs'] = 1
    
    return info

def analyze_single_run(run_dir: Path) -> Dict:
    """Analyze a single run directory."""
    info = {
        'path': run_dir,
        'status': 'unknown',
        'best_score': None,
        'config': None,
        'has_aggregated': False,
        'num_process_files': 0
    }
    
    # Check for aggregated results
    aggregated_file = run_dir / 'aggregated_results.pkl'
    if aggregated_file.exists():
        info['has_aggregated'] = True
        try:
            with open(aggregated_file, 'rb') as f:
                data = pickle.load(f)
            
            if 'summary' in data:
                summary = data['summary']
                info['best_score'] = summary.get('best_overall_score')
                info['config'] = summary.get('config_name')
                info['status'] = 'success'
            else:
                info['status'] = 'incomplete'
                
        except Exception as e:
            info['status'] = 'corrupted'
            info['error'] = str(e)
    else:
        # Check for process files
        process_files = list(run_dir.glob('process_*_results.pkl'))
        info['num_process_files'] = len(process_files)
        
        if info['num_process_files'] == 0:
            info['status'] = 'failed'
        else:
            # Try to get best score from process files
            best_scores = []
            for process_file in process_files:
                try:
                    with open(process_file, 'rb') as f:
                        data = pickle.load(f)
                    if 'best_overall_score' in data:
                        best_scores.append(data['best_overall_score'])
                except:
                    continue
            
            if best_scores:
                info['best_score'] = max(best_scores)
                info['status'] = 'incomplete'  # Has data but no aggregated results
            else:
                info['status'] = 'failed'
    
    return info

def should_delete_directory(info: Dict, threshold: Optional[int], failed_only: bool) -> Tuple[bool, str]:
    """Determine if a directory should be deleted and why."""
    reasons = []
    
    if failed_only:
        if info['status'] == 'failed':
            return True, "Failed run (no valid results)"
        return False, "Not a failed run"
    
    if info['status'] == 'failed':
        return True, "Failed run (no valid results)"
    
    if info['status'] == 'corrupted':
        return True, "Corrupted results file"
    
    if threshold is not None and info['best_score'] is not None:
        if info['best_score'] < threshold:
            return True, f"Low score ({info['best_score']} < {threshold})"
    
    # Additional criteria for incomplete runs
    if info['status'] == 'incomplete':
        if info['type'] == 'multiple_runs':
            success_rate = info['successful_runs'] / max(info['total_runs'], 1)
            if success_rate < 0.2:  # Less than 20% success rate
                return True, f"Low success rate ({success_rate:.1%})"
        
    return False, "Meets retention criteria"

def main():
    parser = argparse.ArgumentParser(
        description="Clean up GA results directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cleanup_results.py --dry-run                    # Preview all cleanup
  python cleanup_results.py --threshold 900 --dry-run    # Preview removal of scores < 900
  python cleanup_results.py --threshold 900              # Remove scores < 900
  python cleanup_results.py --failed-only --dry-run      # Preview failed runs only
  python cleanup_results.py --keep-recent 3              # Keep only 3 most recent result sets
  python cleanup_results.py --threshold 850 --keep-recent 5  # Combination cleanup
        """
    )
    
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Results directory to clean (default: results)")
    parser.add_argument("--threshold", type=int, default=None,
                       help="Delete runs with best score below this threshold")
    parser.add_argument("--keep-recent", type=int, default=None,
                       help="Keep only N most recent result sets")
    parser.add_argument("--failed-only", action="store_true",
                       help="Only remove completely failed runs")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be deleted without actually deleting")
    parser.add_argument("--force", action="store_true",
                       help="Skip confirmation prompt")
    
    args = parser.parse_args()
    
    results_path = Path(args.results_dir)
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_path}")
        sys.exit(1)
    
    print("🔍 RESULTS DIRECTORY CLEANUP")
    print("=" * 50)
    print(f"Scanning: {results_path.absolute()}")
    
    # Get all result directories
    result_dirs = [d for d in results_path.iterdir() 
                   if d.is_dir() and not d.name.startswith('.')]
    
    if not result_dirs:
        print("No result directories found.")
        return
    
    print(f"Found {len(result_dirs)} result directories")
    print("\n🔬 Analyzing results...")
    
    # Analyze each directory
    all_info = []
    total_size = 0
    
    for result_dir in result_dirs:
        print(f"  Analyzing {result_dir.name}...", end=" ")
        info = analyze_result_directory(result_dir)
        all_info.append(info)
        total_size += info['size_bytes']
        print(f"({format_size(info['size_bytes'])})")
    
    print(f"\n📊 Total space used: {format_size(total_size)}")
    
    # Sort by timestamp (most recent first)
    all_info.sort(key=lambda x: x['timestamp'] or datetime.min, reverse=True)
    
    # Determine what to delete
    to_delete = []
    to_keep = []
    
    for i, info in enumerate(all_info):
        should_delete = False
        reason = ""
        
        # Apply keep-recent filter first
        if args.keep_recent is not None and i >= args.keep_recent:
            should_delete = True
            reason = f"Older than {args.keep_recent} most recent"
        else:
            should_delete, reason = should_delete_directory(info, args.threshold, args.failed_only)
        
        if should_delete:
            to_delete.append((info, reason))
        else:
            to_keep.append(info)
    
    # Show summary
    print("\n📋 CLEANUP SUMMARY")
    print("=" * 50)
    
    if not to_delete:
        print("✅ No directories need cleanup!")
        return
    
    delete_size = sum(info['size_bytes'] for info, _ in to_delete)
    keep_size = sum(info['size_bytes'] for info in to_keep)
    
    print(f"Directories to delete: {len(to_delete)}")
    print(f"Directories to keep: {len(to_keep)}")
    print(f"Space to free: {format_size(delete_size)}")
    print(f"Space remaining: {format_size(keep_size)}")
    print(f"Cleanup percentage: {delete_size/total_size*100:.1f}%")
    
    print("\n🗑️ DIRECTORIES TO DELETE:")
    print("-" * 50)
    for info, reason in to_delete:
        status_emoji = {"success": "✅", "failed": "❌", "incomplete": "⚠️", "corrupted": "💥"}.get(info['status'], "❓")
        score_str = f"(Score: {info['best_score']})" if info['best_score'] is not None else "(No score)"
        print(f"{status_emoji} {info['name']} {score_str} - {format_size(info['size_bytes'])} - {reason}")
    
    if to_keep:
        print(f"\n✅ DIRECTORIES TO KEEP ({len(to_keep)}):")
        print("-" * 50)
        for info in to_keep:
            status_emoji = {"success": "✅", "failed": "❌", "incomplete": "⚠️", "corrupted": "💥"}.get(info['status'], "❓")
            score_str = f"(Score: {info['best_score']})" if info['best_score'] is not None else "(No score)"
            date_str = info['timestamp'].strftime('%Y-%m-%d %H:%M') if info['timestamp'] else 'Unknown date'
            print(f"{status_emoji} {info['name']} {score_str} - {format_size(info['size_bytes'])} - {date_str}")
    
    if args.dry_run:
        print(f"\n🔍 DRY RUN COMPLETE - No files were deleted")
        print(f"💾 Would free {format_size(delete_size)} of disk space")
        return
    
    # Confirmation prompt
    if not args.force:
        print(f"\n⚠️ This will permanently delete {len(to_delete)} directories and free {format_size(delete_size)}")
        response = input("Continue? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Cleanup cancelled.")
            return
    
    # Perform deletion
    print(f"\n🗑️ Deleting {len(to_delete)} directories...")
    deleted_count = 0
    freed_space = 0
    
    for info, reason in to_delete:
        try:
            print(f"  Deleting {info['name']}...", end=" ")
            shutil.rmtree(info['path'])
            deleted_count += 1
            freed_space += info['size_bytes']
            print("✅")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n🎉 CLEANUP COMPLETE!")
    print(f"✅ Deleted: {deleted_count}/{len(to_delete)} directories")
    print(f"💾 Freed space: {format_size(freed_space)}")
    print(f"📊 Remaining: {len(to_keep)} directories ({format_size(keep_size)})")

if __name__ == "__main__":
    main()