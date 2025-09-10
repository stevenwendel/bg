#!/usr/bin/env python3
"""
DNA Analysis Module

Handles data loading, filtering, and preprocessing for the analysis notebook.
"""

import os
import pickle
import numpy as np
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import gc

def find_all_aggregated_results(results_folder):
    """Find all aggregated_results.pkl files in the results folder and subfolders, plus cleaned_results."""
    results_path = Path(results_folder)
    
    if not results_path.exists():
        print(f"❌ Results folder does not exist: {results_folder}")
        return []
    
    # Find aggregated_results.pkl files in the main results folder
    aggregated_files = list(results_path.rglob("aggregated_results.pkl"))
    
    # Also search for .pkl files in cleaned_results folder
    cleaned_results_path = Path("cleaned_results")
    if cleaned_results_path.exists():
        cleaned_pkl_files = list(cleaned_results_path.rglob("*.pkl"))
        print(f"📁 Found {len(cleaned_pkl_files)} .pkl files in cleaned_results folder:")
        for f in cleaned_pkl_files:
            print(f"  {f}")
        aggregated_files.extend(cleaned_pkl_files)
    else:
        print("📁 No cleaned_results folder found")
    
    print(f"📁 Found {len(aggregated_files)} total .pkl files")
    print("📁 Main results files:")
    main_files = [f for f in aggregated_files if not str(f).startswith("cleaned_results")]
    for f in main_files:
        print(f"  {f}")
    
    return aggregated_files

def load_single_file(file_path_info):
    """Load a single pickle file and extract high-scoring DNAs."""
    file_path, score_threshold = file_path_info
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        run_folder = file_path.parent.name
        high_scoring_dnas = []
        
        # Handle different file formats
        if isinstance(data, dict):
            # Standard aggregated_results.pkl format
            if 'all_dna_tested' in data:
                dna_records = data['all_dna_tested']
            # Alternative formats from cleaned_results
            elif 'best_dnas' in data:
                dna_records = data['best_dnas']
            elif 'target_dnas' in data:
                dna_records = data['target_dnas']
            # Single DNA record
            elif 'dna' in data and 'total_score' in data:
                dna_records = [data]
            else:
                # Try to find any list-like data that might contain DNAs
                dna_records = []
                for key, value in data.items():
                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        if 'dna' in value[0] and ('total_score' in value[0] or 'score' in value[0]):
                            dna_records = value
                            break
        elif isinstance(data, list):
            # Direct list of DNA records
            dna_records = data
        else:
            print(f"⚠️  Unknown data format in {file_path}")
            return []
        
        # Extract high-scoring DNAs
        for dna_record in dna_records:
            if not isinstance(dna_record, dict):
                continue
                
            # Handle different score field names
            total_score = dna_record.get('total_score')
            if total_score is None:
                total_score = dna_record.get('score')
            if total_score is None:
                total_score = dna_record.get('pruned_score')
            if total_score is None:
                total_score = 0
            
            if total_score >= score_threshold:
                # Handle different DNA field names
                dna_array = dna_record.get('dna')
                if dna_array is None:
                    dna_array = dna_record.get('dna_vector')
                if dna_array is None:
                    dna_array = dna_record.get('pruned_dna')
                
                if dna_array is None:
                    continue
                
                # Convert to numpy array if needed
                if not isinstance(dna_array, np.ndarray):
                    dna_array = np.array(dna_array)
                
                # Extract scores with fallbacks
                exp_score = dna_record.get('exp_score', 0)
                if exp_score == 0:
                    exp_score = dna_record.get('final_exp_score', 0)
                cont_score = dna_record.get('cont_score', 0)
                if cont_score == 0:
                    cont_score = dna_record.get('final_cont_score', 0)
                
                high_scoring_dnas.append({
                    'dna': dna_array,
                    'total_score': total_score,
                    'exp_score': exp_score,
                    'cont_score': cont_score,
                    'generation': dna_record.get('generation', 0),
                    'process_id': dna_record.get('process_id', 0),
                    'individual_id': dna_record.get('individual_id', 0),
                    'run_folder': run_folder,
                    'source_file': str(file_path),
                    'non_zero_weights': int(np.count_nonzero(dna_array)),
                    'dna_hash': hashlib.md5(dna_array.tobytes()).hexdigest()
                })
        
        return high_scoring_dnas
        
    except Exception as e:
        print(f"⚠️  Error reading {file_path}: {e}")
        return []

def remove_exact_duplicates_fast(high_scoring_dnas):
    """Remove DNAs with identical vectors using hash-based deduplication."""
    print(f"🔍 Removing exact duplicates from {len(high_scoring_dnas)} DNAs...")
    
    seen_hashes = set()
    unique_dnas = []
    
    for dna_info in high_scoring_dnas:
        dna_hash = dna_info['dna_hash']
        
        if dna_hash not in seen_hashes:
            seen_hashes.add(dna_hash)
            dna_info['dna'] = dna_info['dna'].copy()
            unique_dnas.append(dna_info)
    
    duplicates_removed = len(high_scoring_dnas) - len(unique_dnas)
    print(f"  ✅ Removed {duplicates_removed} exact duplicates, {len(unique_dnas)} unique DNAs remain")
    
    return unique_dnas

def filter_unique_configurations_fast(high_scoring_dnas):
    """Keep only the best DNA for each unique non-zero weight pattern."""
    print(f"🎯 Filtering for unique configurations from {len(high_scoring_dnas)} DNAs...")
    
    configuration_groups = {}
    
    for dna_info in high_scoring_dnas:
        mask = tuple((dna_info['dna'] != 0).astype(np.uint8))
        
        if mask not in configuration_groups:
            configuration_groups[mask] = []
        
        configuration_groups[mask].append(dna_info)
    
    print(f"  Found {len(configuration_groups)} unique weight configurations:")
    
    unique_config_dnas = []
    
    for i, (mask, dnas_in_group) in enumerate(configuration_groups.items()):
        scores = np.array([d['total_score'] for d in dnas_in_group])
        best_idx = np.argmax(scores)
        best_dna = dnas_in_group[best_idx]
        unique_config_dnas.append(best_dna)
        
        non_zero_count = np.sum(mask)
        
        print(f"    Config {i+1}: {non_zero_count} non-zero weights, "
              f"{len(dnas_in_group)} DNAs (scores: {scores.min()}-{scores.max()}), "
              f"kept best: {best_dna['total_score']}")
    
    scores = np.array([d['total_score'] for d in unique_config_dnas])
    sort_indices = np.argsort(scores)[::-1]
    unique_config_dnas = [unique_config_dnas[i] for i in sort_indices]
    
    configurations_removed = len(high_scoring_dnas) - len(unique_config_dnas)
    print(f"  ✅ Removed {configurations_removed} duplicate configurations, "
          f"{len(unique_config_dnas)} unique configurations remain")
    
    return unique_config_dnas

def extract_high_scoring_dnas(aggregated_files, score_threshold, remove_duplicates=True, unique_configs_only=True):
    """Extract all DNA vectors that exceed the score threshold."""
    print(f"🚀 Loading {len(aggregated_files)} files in parallel...")
    
    file_args = [(file_path, score_threshold) for file_path in aggregated_files]
    
    with ThreadPoolExecutor(max_workers=min(4, len(aggregated_files))) as executor:
        results = list(executor.map(load_single_file, file_args))
    
    high_scoring_dnas = []
    for file_results in results:
        high_scoring_dnas.extend(file_results)
    
    print(f"\n🎯 Found {len(high_scoring_dnas)} DNA vectors with score >= {score_threshold}")
    
    if not high_scoring_dnas:
        return []
    
    if remove_duplicates:
        high_scoring_dnas = remove_exact_duplicates_fast(high_scoring_dnas)
    
    if unique_configs_only:
        high_scoring_dnas = filter_unique_configurations_fast(high_scoring_dnas)
    
    if high_scoring_dnas:
        scores = np.array([d['total_score'] for d in high_scoring_dnas])
        sort_indices = np.argsort(scores)[::-1]
        high_scoring_dnas = [high_scoring_dnas[i] for i in sort_indices]
        
        best_score = high_scoring_dnas[0]['total_score']
        worst_score = high_scoring_dnas[-1]['total_score']
        avg_score = scores.mean()
        
        print(f"\n📊 Final dataset summary:")
        print(f"  Score range: {worst_score} - {best_score}")
        print(f"  Average score: {avg_score:.1f}")
        
        weights = np.array([d['non_zero_weights'] for d in high_scoring_dnas])
        print(f"  Non-zero weights range: {weights.min()} - {weights.max()}")
        
        print(f"\n🏆 Top 5 DNA vectors:")
        for i, dna in enumerate(high_scoring_dnas[:5]):
            print(f"  {i+1}. Score: {dna['total_score']} (Exp:{dna['exp_score']}, Cont:{dna['cont_score']}) "
                  f"Weights:{dna['non_zero_weights']}, Gen:{dna['generation']}, Run: {dna['run_folder']}")
    
    gc.collect()
    return high_scoring_dnas