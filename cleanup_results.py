#!/usr/bin/env python3
"""
Results Folder Cleanup Script

This script scans through a results folder, extracts DNA objects that exceed a given 
score threshold, removes duplicates, and creates a new clean dataset while deleting 
the old data to save space.

Usage:
    python cleanup_results.py --results-dir results/ --threshold 970 --output-dir cleaned_results/
    python cleanup_results.py --threshold 900 --dry-run  # Preview without changes
"""

import os
import pickle
import argparse
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from datetime import datetime

def calculate_dna_hash(dna_array):
    """Calculate MD5 hash of DNA array for deduplication."""
    return hashlib.md5(dna_array.tobytes()).hexdigest()

def scan_pickle_file(pickle_path, threshold):
    """
    Scan a single pickle file and extract high-scoring DNA objects.
    
    Returns:
        List of high-scoring DNA records with metadata
    """
    high_scoring_dnas = []
    
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract run info from path
        run_folder = pickle_path.parent.name
        
        # Handle different pickle file formats
        dna_records = []
        
        if isinstance(data, dict):
            if 'all_dna_tested' in data:
                # Standard aggregated_results.pkl format
                dna_records = data['all_dna_tested']
            elif 'dna' in data and 'total_score' in data:
                # Single DNA record format
                dna_records = [data]
            elif 'final_population' in data:
                # Population format
                dna_records = data['final_population']
        elif isinstance(data, list):
            # List of DNA records
            dna_records = data
        
        # Process each DNA record
        for record in dna_records:
            try:
                # Extract DNA array and score
                if isinstance(record, dict):
                    if 'dna' in record:
                        dna_array = record['dna']
                        if 'total_score' in record:
                            total_score = record['total_score']
                        elif 'dna_score' in record:
                            total_score = record['dna_score']
                        else:
                            continue
                    else:
                        continue
                else:
                    continue
                
                # Check if score exceeds threshold
                if total_score >= threshold:
                    # Create standardized record
                    clean_record = {
                        'dna': np.array(dna_array, dtype=np.int32),
                        'total_score': int(total_score),
                        'exp_score': record.get('exp_score', 0),
                        'cont_score': record.get('cont_score', 0),
                        'generation': record.get('generation', 0),
                        'process_id': record.get('process_id', 0),
                        'individual_id': record.get('individual_id', 0),
                        'source_file': str(pickle_path),
                        'run_folder': run_folder,
                        'dna_hash': calculate_dna_hash(np.array(dna_array, dtype=np.int32)),
                        'non_zero_weights': int(np.count_nonzero(dna_array)),
                        'timestamp': record.get('timestamp', 0)
                    }
                    
                    high_scoring_dnas.append(clean_record)
                    
            except Exception as e:
                print(f"    ⚠️  Error processing DNA record in {pickle_path}: {e}")
                continue
                
    except Exception as e:
        print(f"    ❌ Error reading {pickle_path}: {e}")
        return []
    
    return high_scoring_dnas

def scan_results_folder(results_dir, threshold):
    """
    Scan entire results folder for high-scoring DNAs.
    
    Returns:
        List of all high-scoring DNA records
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ Results directory does not exist: {results_dir}")
        return []
    
    print(f"🔍 Scanning {results_dir} for pickle files...")
    
    # Find all pickle files
    pickle_files = list(results_path.rglob("*.pkl"))
    print(f"📁 Found {len(pickle_files)} pickle files")
    
    all_high_scoring = []
    total_files_processed = 0
    
    for pickle_file in pickle_files:
        print(f"  📄 Processing {pickle_file}...")
        high_scoring = scan_pickle_file(pickle_file, threshold)
        
        if high_scoring:
            print(f"    ✅ Found {len(high_scoring)} high-scoring DNAs")
            all_high_scoring.extend(high_scoring)
        else:
            print(f"    ⚪ No high-scoring DNAs found")
            
        total_files_processed += 1
    
    print(f"\n📊 Scan complete:")
    print(f"  Files processed: {total_files_processed}")
    print(f"  Total high-scoring DNAs found: {len(all_high_scoring)}")
    
    return all_high_scoring

def remove_duplicates(high_scoring_dnas):
    """
    Remove duplicate DNA vectors using hash-based deduplication.
    Keeps the highest-scoring version of each unique DNA.
    
    Returns:
        List of unique DNA records
    """
    print(f"\n🔍 Removing duplicates from {len(high_scoring_dnas)} DNA records...")
    
    # Group by DNA hash
    hash_groups = defaultdict(list)
    for dna_record in high_scoring_dnas:
        hash_groups[dna_record['dna_hash']].append(dna_record)
    
    unique_dnas = []
    duplicates_removed = 0
    
    for dna_hash, group in hash_groups.items():
        if len(group) == 1:
            # No duplicates
            unique_dnas.append(group[0])
        else:
            # Multiple copies - keep the highest scoring one
            best_record = max(group, key=lambda x: x['total_score'])
            unique_dnas.append(best_record)
            duplicates_removed += len(group) - 1
            
            print(f"  🔗 Hash {dna_hash[:8]}... had {len(group)} copies, kept best (score: {best_record['total_score']})")
    
    print(f"  ✅ Removed {duplicates_removed} duplicates")
    print(f"  ✅ {len(unique_dnas)} unique DNA records remain")
    
    return unique_dnas

def create_cleaned_dataset(unique_dnas, output_dir, threshold):
    """
    Create new cleaned dataset with high-scoring unique DNAs.
    
    Returns:
        Path to the created file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"cleaned_high_scoring_dnas_{timestamp}.pkl"
    
    # Sort by score (highest first)
    unique_dnas.sort(key=lambda x: x['total_score'], reverse=True)
    
    # Create summary statistics
    scores = [dna['total_score'] for dna in unique_dnas]
    weights = [dna['non_zero_weights'] for dna in unique_dnas]
    
    cleaned_dataset = {
        'metadata': {
            'created_timestamp': timestamp,
            'total_dnas': len(unique_dnas),
            'score_threshold_used': threshold,
            'score_range': {
                'min': min(scores) if scores else 0,
                'max': max(scores) if scores else 0,
                'mean': np.mean(scores) if scores else 0
            },
            'weight_range': {
                'min': min(weights) if weights else 0,
                'max': max(weights) if weights else 0,
                'mean': np.mean(weights) if weights else 0
            },
            'source_info': {
                'unique_run_folders': list(set(dna['run_folder'] for dna in unique_dnas)),
                'generation_range': {
                    'min': min(dna['generation'] for dna in unique_dnas) if unique_dnas else 0,
                    'max': max(dna['generation'] for dna in unique_dnas) if unique_dnas else 0
                }
            }
        },
        'high_scoring_dnas': unique_dnas
    }
    
    # Save cleaned dataset
    with open(output_file, 'wb') as f:
        pickle.dump(cleaned_dataset, f)
    
    print(f"\n💾 Cleaned dataset saved to: {output_file}")
    print(f"📊 Dataset contains {len(unique_dnas)} unique high-scoring DNAs")
    
    if unique_dnas:
        print(f"   Score range: {min(scores)} - {max(scores)} (avg: {np.mean(scores):.1f})")
        print(f"   Weight range: {min(weights)} - {max(weights)} (avg: {np.mean(weights):.1f})")
        print(f"   Source runs: {len(set(dna['run_folder'] for dna in unique_dnas))}")
        
        # Show top 5
        print(f"\n🏆 Top 5 DNAs:")
        for i, dna in enumerate(unique_dnas[:5]):
            print(f"   {i+1}. Score: {dna['total_score']}, Weights: {dna['non_zero_weights']}, "
                  f"Run: {dna['run_folder']}, Gen: {dna['generation']}")
    
    return output_file

def delete_old_data(results_dir, dry_run=False):
    """
    Delete the original results folder to free up space.
    """
    if dry_run:
        print(f"\n🔍 DRY RUN: Would delete {results_dir}")
        return
    
    print(f"\n🗑️  Deleting original results folder: {results_dir}")
    
    try:
        shutil.rmtree(results_dir)
        print(f"   ✅ Successfully deleted {results_dir}")
    except Exception as e:
        print(f"   ❌ Error deleting {results_dir}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Clean up results folder by extracting high-scoring DNAs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean with default threshold, delete old data
  python cleanup_results.py --results-dir results/ --threshold 970
  
  # Preview what would be cleaned without making changes
  python cleanup_results.py --results-dir results/ --threshold 900 --dry-run
  
  # Specify custom output directory
  python cleanup_results.py --results-dir results/ --output-dir my_cleaned_results/ --threshold 950
        """
    )
    
    parser.add_argument("--results-dir", default="results", 
                       help="Path to results folder to clean (default: results)")
    parser.add_argument("--output-dir", default="cleaned_results",
                       help="Path to output cleaned dataset (default: cleaned_results)")
    parser.add_argument("--threshold", type=int, default=970,
                       help="Minimum score threshold for DNA selection (default: 970)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Preview actions without making changes")
    parser.add_argument("--keep-original", action="store_true",
                       help="Keep original results folder (don't delete)")
    
    args = parser.parse_args()
    
    print("🧹 RESULTS FOLDER CLEANUP TOOL")
    print("=" * 50)
    print(f"Input folder: {args.results_dir}")
    print(f"Output folder: {args.output_dir}")
    print(f"Score threshold: {args.threshold}")
    print(f"Mode: {'DRY RUN (preview only)' if args.dry_run else 'LIVE (will make changes)'}")
    print("=" * 50)
    
    # Step 1: Scan for high-scoring DNAs
    high_scoring_dnas = scan_results_folder(args.results_dir, args.threshold)
    
    if not high_scoring_dnas:
        print(f"\n❌ No DNA vectors found with score >= {args.threshold}")
        return
    
    print(f"\n✅ Found {len(high_scoring_dnas)} DNA vectors with score >= {args.threshold}")
    
    # Step 2: Remove duplicates
    unique_dnas = remove_duplicates(high_scoring_dnas)
    
    if args.dry_run:
        print(f"\n🔍 DRY RUN SUMMARY:")
        print(f"   Would save {len(unique_dnas)} unique high-scoring DNAs")
        print(f"   Would create cleaned dataset in: {args.output_dir}")
        if not args.keep_original:
            print(f"   Would delete original folder: {args.results_dir}")
        print(f"\n💡 Run without --dry-run to execute cleanup")
        return
    
    # Step 3: Create cleaned dataset
    output_file = create_cleaned_dataset(unique_dnas, args.output_dir, args.threshold)
    
    # Step 4: Delete old data (unless keeping original)
    if not args.keep_original:
        response = input(f"\n⚠️  Delete original results folder '{args.results_dir}'? [y/N]: ")
        if response.lower().strip() in ['y', 'yes']:
            delete_old_data(args.results_dir, dry_run=False)
        else:
            print(f"   ℹ️  Original folder '{args.results_dir}' kept")
    
    print(f"\n🎉 Cleanup complete!")
    print(f"   Cleaned dataset: {output_file}")
    print(f"   DNA vectors saved: {len(unique_dnas)}")
    
    # Calculate space savings estimate
    try:
        original_size_mb = sum(f.stat().st_size for f in Path(args.results_dir).rglob('*') if f.is_file()) / 1024 / 1024
        new_size_mb = Path(output_file).stat().st_size / 1024 / 1024
        
        print(f"   Estimated space saved: {original_size_mb:.1f}MB → {new_size_mb:.1f}MB "
              f"({((original_size_mb - new_size_mb) / original_size_mb * 100):.1f}% reduction)")
    except:
        print("   Could not calculate space savings")

if __name__ == "__main__":
    main()