#!/usr/bin/env python3
"""
Simple Results Cleanup Script

Focuses on quick wins for disk space cleanup:
1. Remove directories with no aggregated_results.pkl files
2. Remove directories by score threshold (quick check)
3. Show directory sizes for manual cleanup

Usage:
    python simple_cleanup.py                              # Show overview
    python simple_cleanup.py --remove-failed              # Remove obvious failures  
    python simple_cleanup.py --threshold 800 --dry-run    # Preview score-based cleanup
"""

import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path
import subprocess

def get_size_mb(path):
    """Get directory size in MB using du command (fast)."""
    try:
        result = subprocess.run(['du', '-sm', str(path)], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return int(result.stdout.split()[0])
    except:
        pass
    return 0

def check_directory_quick(dir_path):
    """Quick check of directory status."""
    info = {
        'path': dir_path,
        'name': dir_path.name,
        'size_mb': get_size_mb(dir_path),
        'has_results': False,
        'best_score': None,
        'status': 'unknown'
    }
    
    # Check for any aggregated_results.pkl files
    aggregated_files = list(dir_path.rglob('aggregated_results.pkl'))
    
    if not aggregated_files:
        info['status'] = 'no_results'
        return info
    
    info['has_results'] = True
    
    # Try to get best score from first aggregated file (quick check)
    try:
        with open(aggregated_files[0], 'rb') as f:
            data = pickle.load(f)
        
        if 'summary' in data and 'best_overall_score' in data['summary']:
            info['best_score'] = data['summary']['best_overall_score']
            info['status'] = 'success'
        else:
            info['status'] = 'incomplete'
    except:
        info['status'] = 'corrupted'
    
    return info

def main():
    parser = argparse.ArgumentParser(description="Simple results cleanup")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--remove-failed", action="store_true", help="Remove directories with no results")
    parser.add_argument("--threshold", type=int, help="Remove runs below this score")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--force", action="store_true", help="Skip confirmation")
    
    args = parser.parse_args()
    
    results_path = Path(args.results_dir)
    if not results_path.exists():
        print(f"❌ Directory not found: {results_path}")
        sys.exit(1)
    
    print("📁 SIMPLE RESULTS CLEANUP")
    print("=" * 40)
    
    # Get directories
    result_dirs = [d for d in results_path.iterdir() 
                   if d.is_dir() and not d.name.startswith('.')]
    
    if not result_dirs:
        print("No directories found.")
        return
    
    print(f"Checking {len(result_dirs)} directories...")
    
    # Quick analysis
    all_info = []
    total_mb = 0
    
    for dir_path in result_dirs:
        print(f"  Checking {dir_path.name}...", end="")
        info = check_directory_quick(dir_path)
        all_info.append(info)
        total_mb += info['size_mb']
        
        status_emoji = {
            'success': '✅',
            'no_results': '❌',
            'corrupted': '💥',
            'incomplete': '⚠️'
        }.get(info['status'], '❓')
        
        score_str = f" ({info['best_score']})" if info['best_score'] else ""
        print(f" {status_emoji}{score_str} {info['size_mb']}MB")
    
    print(f"\nTotal space: {total_mb}MB ({total_mb/1024:.1f}GB)")
    
    # Categorize
    no_results = [info for info in all_info if info['status'] == 'no_results']
    low_scores = []
    if args.threshold:
        low_scores = [info for info in all_info 
                     if info['best_score'] is not None and info['best_score'] < args.threshold]
    
    # Show summary
    print(f"\n📊 SUMMARY")
    print("-" * 40)
    print(f"Total directories: {len(all_info)}")
    print(f"No results files: {len(no_results)} ({sum(i['size_mb'] for i in no_results)}MB)")
    if args.threshold:
        print(f"Below threshold {args.threshold}: {len(low_scores)} ({sum(i['size_mb'] for i in low_scores)}MB)")
    
    # Determine what to delete
    to_delete = []
    
    if args.remove_failed:
        to_delete.extend([(info, 'No results files') for info in no_results])
    
    if args.threshold:
        to_delete.extend([(info, f'Score {info["best_score"]} < {args.threshold}') 
                         for info in low_scores])
    
    if not to_delete:
        if not (args.remove_failed or args.threshold):
            # Just show overview
            print(f"\n📋 DIRECTORY OVERVIEW (sorted by size):")
            sorted_info = sorted(all_info, key=lambda x: x['size_mb'], reverse=True)
            for info in sorted_info:
                status_emoji = {
                    'success': '✅',
                    'no_results': '❌',
                    'corrupted': '💥',
                    'incomplete': '⚠️'
                }.get(info['status'], '❓')
                score_str = f" (Score: {info['best_score']})" if info['best_score'] else ""
                print(f"  {status_emoji} {info['name']}{score_str} - {info['size_mb']}MB")
            
            print(f"\n💡 CLEANUP SUGGESTIONS:")
            if no_results:
                print(f"   --remove-failed would free {sum(i['size_mb'] for i in no_results)}MB from {len(no_results)} failed directories")
            
            # Suggest thresholds
            scores = [info['best_score'] for info in all_info if info['best_score'] is not None]
            if scores:
                scores.sort()
                if len(scores) > 5:
                    low_threshold = scores[len(scores)//3]  # Bottom third
                    mid_threshold = scores[len(scores)//2]  # Median
                    print(f"   --threshold {low_threshold} would remove bottom third of results")
                    print(f"   --threshold {mid_threshold} would remove bottom half of results")
        else:
            print("✅ No directories match deletion criteria!")
        return
    
    # Show deletion plan
    delete_mb = sum(info['size_mb'] for info, _ in to_delete)
    print(f"\n🗑️ DELETION PLAN:")
    print(f"Directories to delete: {len(to_delete)}")
    print(f"Space to free: {delete_mb}MB ({delete_mb/1024:.1f}GB)")
    print(f"Remaining: {(total_mb-delete_mb)}MB ({(total_mb-delete_mb)/1024:.1f}GB)")
    
    print(f"\nDirectories to delete:")
    for info, reason in to_delete:
        print(f"  ❌ {info['name']} - {info['size_mb']}MB - {reason}")
    
    if args.dry_run:
        print(f"\n🔍 DRY RUN - Would free {delete_mb}MB")
        return
    
    # Confirm deletion
    if not args.force:
        print(f"\n⚠️ This will delete {len(to_delete)} directories ({delete_mb}MB)")
        if input("Continue? (y/N): ").lower() not in ['y', 'yes']:
            print("Cancelled.")
            return
    
    # Delete
    print(f"\n🗑️ Deleting directories...")
    deleted = 0
    freed_mb = 0
    
    for info, reason in to_delete:
        try:
            print(f"  Deleting {info['name']} ({info['size_mb']}MB)...", end="")
            shutil.rmtree(info['path'])
            deleted += 1
            freed_mb += info['size_mb']
            print(" ✅")
        except Exception as e:
            print(f" ❌ Error: {e}")
    
    print(f"\n🎉 CLEANUP COMPLETE!")
    print(f"✅ Deleted: {deleted}/{len(to_delete)} directories")
    print(f"💾 Freed: {freed_mb}MB ({freed_mb/1024:.1f}GB)")

if __name__ == "__main__":
    main()