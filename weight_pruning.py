#!/usr/bin/env python3
"""
Weight Pruning Algorithm for Neural Network DNA Optimization

This script implements algorithms to find the minimal DNA vector by systematically
removing weights while maintaining performance above a threshold.

Strategies implemented:
1. Greedy removal: Remove smallest absolute weights first
2. Impact-based removal: Remove weights with least impact on score
3. Iterative removal: Remove one weight at a time, keeping best result
4. Group removal: Remove multiple similar weights simultaneously
"""

import pickle
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse
from copy import deepcopy

# Import necessary components from the GA system
from src.constants import *
from adaptive_tmax_fully_optimized import (
    evaluate_population_fully_optimized, 
    initialize_connection_mapping,
    get_cue_go_waves_for_tmax,
    get_criteria_for_tmax,
    alpha, td, ALPHA_L
)

# Global variables for evaluation
conn_map = initialize_connection_mapping(ACTIVE_SYNAPSES, NEURON_NAMES)
N = len(NEURON_NAMES)

def evaluate_single_dna(dna_vector: np.ndarray, tmax: int = 5000) -> Tuple[int, int, int]:
    """
    Evaluate a single DNA vector and return (exp_score, cont_score, total_score).
    
    Args:
        dna_vector: DNA vector to evaluate
        tmax: Simulation time (default: 5000)
    
    Returns:
        Tuple of (experimental_score, control_score, total_score)
    """
    # Get cue/go waves and criteria for the specified TMAX
    cue_wave, go_wave = get_cue_go_waves_for_tmax(tmax)
    crit_Exp_trunc, crit_Cont_trunc, crit_indices_fixed, pass_ids_fixed = get_criteria_for_tmax(tmax)
    
    # Evaluate the single DNA vector (wrap in array for batch processing)
    scores = evaluate_population_fully_optimized(
        np.array([dna_vector]), conn_map, N, a, b, vreset, d, k, vr, vt, vpeak, C, E,
        alpha, cue_wave, go_wave, crit_Exp_trunc, crit_Cont_trunc, 
        crit_indices_fixed, pass_ids_fixed, tmax, 
        batch_size=1, early_termination_threshold=0, use_reduced_precision=False
    )
    
    _, exp_score, cont_score = scores[0]
    return int(exp_score), int(cont_score), int(exp_score + cont_score)


class WeightPruner:
    """Class for pruning weights from DNA vectors while maintaining performance."""
    
    def __init__(self, original_dna: np.ndarray, min_score_threshold: int = 970, tmax: int = 5000):
        """
        Initialize the weight pruner.
        
        Args:
            original_dna: Starting DNA vector
            min_score_threshold: Minimum acceptable total score
            tmax: Simulation time for evaluation
        """
        self.original_dna = original_dna.copy()
        self.min_score_threshold = min_score_threshold
        self.tmax = tmax
        
        # Evaluate original DNA
        self.original_exp, self.original_cont, self.original_total = evaluate_single_dna(
            self.original_dna, self.tmax
        )
        
        print(f"Original DNA evaluation:")
        print(f"  Exp: {self.original_exp}, Cont: {self.original_cont}, Total: {self.original_total}")
        print(f"  Non-zero weights: {np.count_nonzero(self.original_dna)}")
        print(f"  Zero weights: {len(self.original_dna) - np.count_nonzero(self.original_dna)}")
        
        # Track pruning history
        self.pruning_history = []
        
    def greedy_magnitude_pruning(self, max_removals: Optional[int] = None) -> np.ndarray:
        """
        Remove weights by smallest absolute magnitude first.
        
        Args:
            max_removals: Maximum number of weights to remove (None for unlimited)
            
        Returns:
            Optimally pruned DNA vector
        """
        print(f"\n🔍 Starting greedy magnitude pruning (threshold: {self.min_score_threshold})")
        
        current_dna = self.original_dna.copy()
        removals_made = 0
        
        # Get indices of non-zero weights sorted by absolute magnitude
        nonzero_indices = np.where(current_dna != 0)[0]
        sorted_indices = sorted(nonzero_indices, key=lambda i: abs(current_dna[i]))
        
        for i, weight_idx in enumerate(sorted_indices):
            if max_removals and removals_made >= max_removals:
                break
                
            # Try removing this weight
            test_dna = current_dna.copy()
            original_value = test_dna[weight_idx]
            test_dna[weight_idx] = 0
            
            # Evaluate the modified DNA
            exp_score, cont_score, total_score = evaluate_single_dna(test_dna, self.tmax)
            
            if total_score >= self.min_score_threshold:
                # Removal successful
                current_dna[weight_idx] = 0
                removals_made += 1
                
                self.pruning_history.append({
                    'method': 'greedy_magnitude',
                    'weight_idx': weight_idx,
                    'original_value': original_value,
                    'new_score': total_score,
                    'exp_score': exp_score,
                    'cont_score': cont_score,
                    'nonzero_count': np.count_nonzero(current_dna)
                })
                
                print(f"  ✅ Removed weight {weight_idx} (value: {original_value:3d}) -> "
                      f"Score: {total_score} (Exp: {exp_score}, Cont: {cont_score}), "
                      f"Non-zero: {np.count_nonzero(current_dna)}")
            else:
                print(f"  ❌ Cannot remove weight {weight_idx} (value: {original_value:3d}) -> "
                      f"Score: {total_score} < {self.min_score_threshold}")
        
        final_nonzero = np.count_nonzero(current_dna)
        print(f"\n✅ Greedy magnitude pruning complete:")
        print(f"  Weights removed: {removals_made}")
        print(f"  Final non-zero weights: {final_nonzero}")
        print(f"  Reduction: {np.count_nonzero(self.original_dna) - final_nonzero} weights")
        
        return current_dna
    
    def impact_based_pruning(self, max_removals: Optional[int] = None) -> np.ndarray:
        """
        Remove weights based on their impact on the score.
        
        Args:
            max_removals: Maximum number of weights to remove
            
        Returns:
            Optimally pruned DNA vector
        """
        print(f"\n🎯 Starting impact-based pruning (threshold: {self.min_score_threshold})")
        
        current_dna = self.original_dna.copy()
        removals_made = 0
        
        while True:
            if max_removals and removals_made >= max_removals:
                break
                
            # Get current non-zero weights
            nonzero_indices = np.where(current_dna != 0)[0]
            
            if len(nonzero_indices) == 0:
                break
            
            best_removal = None
            best_score = -1
            
            # Test removing each non-zero weight
            for weight_idx in nonzero_indices:
                test_dna = current_dna.copy()
                original_value = test_dna[weight_idx]
                test_dna[weight_idx] = 0
                
                exp_score, cont_score, total_score = evaluate_single_dna(test_dna, self.tmax)
                
                if total_score >= self.min_score_threshold and total_score > best_score:
                    best_removal = {
                        'weight_idx': weight_idx,
                        'original_value': original_value,
                        'exp_score': exp_score,
                        'cont_score': cont_score,
                        'total_score': total_score
                    }
                    best_score = total_score
            
            if best_removal is None:
                print("  🛑 No more weights can be removed while maintaining threshold")
                break
            
            # Apply the best removal
            current_dna[best_removal['weight_idx']] = 0
            removals_made += 1
            
            self.pruning_history.append({
                'method': 'impact_based',
                **best_removal,
                'nonzero_count': np.count_nonzero(current_dna)
            })
            
            print(f"  ✅ Removed weight {best_removal['weight_idx']} "
                  f"(value: {best_removal['original_value']:3d}) -> "
                  f"Score: {best_removal['total_score']} "
                  f"(Exp: {best_removal['exp_score']}, Cont: {best_removal['cont_score']}), "
                  f"Non-zero: {np.count_nonzero(current_dna)}")
        
        final_nonzero = np.count_nonzero(current_dna)
        print(f"\n✅ Impact-based pruning complete:")
        print(f"  Weights removed: {removals_made}")
        print(f"  Final non-zero weights: {final_nonzero}")
        print(f"  Reduction: {np.count_nonzero(self.original_dna) - final_nonzero} weights")
        
        return current_dna
    
    def iterative_best_pruning(self, max_removals: Optional[int] = None, 
                              magnitude_first: bool = True) -> np.ndarray:
        """
        Iteratively remove weights, always choosing the best option at each step.
        
        Args:
            max_removals: Maximum number of weights to remove
            magnitude_first: If True, prioritize smaller weights when scores are equal
            
        Returns:
            Optimally pruned DNA vector
        """
        print(f"\n🔄 Starting iterative best pruning (threshold: {self.min_score_threshold})")
        
        current_dna = self.original_dna.copy()
        removals_made = 0
        
        while True:
            if max_removals and removals_made >= max_removals:
                break
                
            nonzero_indices = np.where(current_dna != 0)[0]
            
            if len(nonzero_indices) == 0:
                break
            
            best_candidates = []
            
            # Test removing each non-zero weight
            for weight_idx in nonzero_indices:
                test_dna = current_dna.copy()
                original_value = test_dna[weight_idx]
                test_dna[weight_idx] = 0
                
                exp_score, cont_score, total_score = evaluate_single_dna(test_dna, self.tmax)
                
                if total_score >= self.min_score_threshold:
                    best_candidates.append({
                        'weight_idx': weight_idx,
                        'original_value': original_value,
                        'exp_score': exp_score,
                        'cont_score': cont_score,
                        'total_score': total_score,
                        'abs_value': abs(original_value)
                    })
            
            if not best_candidates:
                print("  🛑 No more weights can be removed while maintaining threshold")
                break
            
            # Choose best candidate
            if magnitude_first:
                # Sort by score (descending), then by absolute value (ascending)
                best_candidates.sort(key=lambda x: (-x['total_score'], x['abs_value']))
            else:
                # Sort by score only (descending)
                best_candidates.sort(key=lambda x: -x['total_score'])
            
            best_removal = best_candidates[0]
            
            # Apply the removal
            current_dna[best_removal['weight_idx']] = 0
            removals_made += 1
            
            self.pruning_history.append({
                'method': 'iterative_best',
                **best_removal,
                'nonzero_count': np.count_nonzero(current_dna)
            })
            
            print(f"  ✅ Removed weight {best_removal['weight_idx']} "
                  f"(value: {best_removal['original_value']:3d}) -> "
                  f"Score: {best_removal['total_score']} "
                  f"(Exp: {best_removal['exp_score']}, Cont: {best_removal['cont_score']}), "
                  f"Non-zero: {np.count_nonzero(current_dna)}")
        
        final_nonzero = np.count_nonzero(current_dna)
        print(f"\n✅ Iterative best pruning complete:")
        print(f"  Weights removed: {removals_made}")
        print(f"  Final non-zero weights: {final_nonzero}")
        print(f"  Reduction: {np.count_nonzero(self.original_dna) - final_nonzero} weights")
        
        return current_dna
    
    def group_removal_pruning(self, group_size: int = 2, max_groups: Optional[int] = None) -> np.ndarray:
        """
        Try removing groups of similar-magnitude weights simultaneously.
        
        Args:
            group_size: Number of weights to remove per group
            max_groups: Maximum number of groups to remove
            
        Returns:
            Optimally pruned DNA vector
        """
        print(f"\n👥 Starting group removal pruning (group_size: {group_size}, threshold: {self.min_score_threshold})")
        
        current_dna = self.original_dna.copy()
        groups_removed = 0
        
        while True:
            if max_groups and groups_removed >= max_groups:
                break
                
            nonzero_indices = np.where(current_dna != 0)[0]
            
            if len(nonzero_indices) < group_size:
                break
            
            # Sort by absolute magnitude
            sorted_indices = sorted(nonzero_indices, key=lambda i: abs(current_dna[i]))
            
            best_group = None
            best_score = -1
            
            # Try different groups of consecutive weights (by magnitude)
            for start_idx in range(len(sorted_indices) - group_size + 1):
                group_indices = sorted_indices[start_idx:start_idx + group_size]
                
                test_dna = current_dna.copy()
                original_values = [test_dna[i] for i in group_indices]
                
                # Remove the group
                for i in group_indices:
                    test_dna[i] = 0
                
                exp_score, cont_score, total_score = evaluate_single_dna(test_dna, self.tmax)
                
                if total_score >= self.min_score_threshold and total_score > best_score:
                    best_group = {
                        'indices': group_indices,
                        'original_values': original_values,
                        'exp_score': exp_score,
                        'cont_score': cont_score,
                        'total_score': total_score
                    }
                    best_score = total_score
            
            if best_group is None:
                print(f"  🛑 No group of {group_size} weights can be removed while maintaining threshold")
                break
            
            # Apply the best group removal
            for i in best_group['indices']:
                current_dna[i] = 0
            
            groups_removed += 1
            
            self.pruning_history.append({
                'method': 'group_removal',
                'group_indices': best_group['indices'],
                'original_values': best_group['original_values'],
                'exp_score': best_group['exp_score'],
                'cont_score': best_group['cont_score'],
                'total_score': best_group['total_score'],
                'nonzero_count': np.count_nonzero(current_dna)
            })
            
            print(f"  ✅ Removed group {best_group['indices']} "
                  f"(values: {best_group['original_values']}) -> "
                  f"Score: {best_group['total_score']} "
                  f"(Exp: {best_group['exp_score']}, Cont: {best_group['cont_score']}), "
                  f"Non-zero: {np.count_nonzero(current_dna)}")
        
        final_nonzero = np.count_nonzero(current_dna)
        print(f"\n✅ Group removal pruning complete:")
        print(f"  Groups removed: {groups_removed}")
        print(f"  Weights removed: {groups_removed * group_size}")
        print(f"  Final non-zero weights: {final_nonzero}")
        print(f"  Reduction: {np.count_nonzero(self.original_dna) - final_nonzero} weights")
        
        return current_dna
    
    def save_results(self, pruned_dna: np.ndarray, method_name: str, output_file: str):
        """Save pruning results to file."""
        final_exp, final_cont, final_total = evaluate_single_dna(pruned_dna, self.tmax)
        
        results = {
            'method': method_name,
            'original_dna': self.original_dna,
            'pruned_dna': pruned_dna,
            'original_scores': {
                'exp': self.original_exp,
                'cont': self.original_cont,
                'total': self.original_total
            },
            'final_scores': {
                'exp': final_exp,
                'cont': final_cont,
                'total': final_total
            },
            'original_nonzero': np.count_nonzero(self.original_dna),
            'final_nonzero': np.count_nonzero(pruned_dna),
            'weights_removed': np.count_nonzero(self.original_dna) - np.count_nonzero(pruned_dna),
            'pruning_history': self.pruning_history,
            'min_score_threshold': self.min_score_threshold,
            'tmax': self.tmax,
            'timestamp': time.time()
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\n💾 Results saved to {output_file}")
        print(f"  Final score: {final_total} (Exp: {final_exp}, Cont: {final_cont})")
        print(f"  Weights reduced: {np.count_nonzero(self.original_dna)} → {np.count_nonzero(pruned_dna)}")
        print(f"  Reduction: {results['weights_removed']} weights removed")


def run_comprehensive_pruning(dna_file: str, min_score: int = 970, tmax: int = 5000, 
                             output_dir: str = "pruning_results"):
    """
    Run all pruning methods on the DNA and compare results.
    
    Args:
        dna_file: Path to pickled DNA file
        min_score: Minimum acceptable score
        tmax: Simulation time
        output_dir: Directory to save results
    """
    # Load DNA
    data = pickle.load(open(dna_file, 'rb'))
    if isinstance(data, dict) and 'dna' in data:
        original_dna = data['dna']
        print(f"Loaded DNA from dict with score: {data.get('total_score', 'Unknown')}")
    else:
        original_dna = data
        print(f"Loaded DNA vector directly")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"🧬 Starting comprehensive weight pruning")
    print(f"  Original DNA length: {len(original_dna)}")
    print(f"  Original non-zero weights: {np.count_nonzero(original_dna)}")
    print(f"  Minimum score threshold: {min_score}")
    print(f"  TMAX: {tmax}")
    print("=" * 80)
    
    # Initialize pruner
    pruner = WeightPruner(original_dna, min_score, tmax)
    
    methods = [
        ("greedy_magnitude", lambda: pruner.greedy_magnitude_pruning()),
        ("impact_based", lambda: pruner.impact_based_pruning()),
        ("iterative_best", lambda: pruner.iterative_best_pruning()),
        ("group_removal_2", lambda: pruner.group_removal_pruning(group_size=2)),
        ("group_removal_3", lambda: pruner.group_removal_pruning(group_size=3)),
    ]
    
    results_summary = []
    
    for method_name, method_func in methods:
        print(f"\n{'='*20} {method_name.upper()} {'='*20}")
        
        # Reset pruning history for each method
        pruner.pruning_history = []
        
        start_time = time.time()
        try:
            pruned_dna = method_func()
            end_time = time.time()
            
            # Save results
            output_file = output_path / f"{method_name}_results.pkl"
            pruner.save_results(pruned_dna, method_name, str(output_file))
            
            # Evaluate final result
            final_exp, final_cont, final_total = evaluate_single_dna(pruned_dna, tmax)
            
            results_summary.append({
                'method': method_name,
                'final_score': final_total,
                'exp_score': final_exp,
                'cont_score': final_cont,
                'final_nonzero': np.count_nonzero(pruned_dna),
                'weights_removed': np.count_nonzero(original_dna) - np.count_nonzero(pruned_dna),
                'time_taken': end_time - start_time,
                'success': final_total >= min_score
            })
            
        except Exception as e:
            print(f"❌ Method {method_name} failed: {e}")
            results_summary.append({
                'method': method_name,
                'success': False,
                'error': str(e)
            })
    
    # Print final comparison
    print(f"\n{'='*80}")
    print("🏆 PRUNING METHODS COMPARISON")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'Score':<6} {'Non-zero':<9} {'Removed':<8} {'Time':<8} {'Success'}")
    print("-" * 80)
    
    for result in results_summary:
        if result['success']:
            print(f"{result['method']:<20} {result['final_score']:<6} "
                  f"{result['final_nonzero']:<9} {result['weights_removed']:<8} "
                  f"{result['time_taken']:<8.1f} ✅")
        else:
            print(f"{result['method']:<20} {'FAIL':<6} {'':<9} {'':<8} {'':<8} ❌")
    
    # Find best method
    successful_results = [r for r in results_summary if r['success']]
    if successful_results:
        best_result = min(successful_results, key=lambda x: x['final_nonzero'])
        print(f"\n🥇 Best method: {best_result['method']}")
        print(f"   Final non-zero weights: {best_result['final_nonzero']}")
        print(f"   Weights removed: {best_result['weights_removed']}")
        print(f"   Final score: {best_result['final_score']}")
    
    # Save summary
    summary_file = output_path / "pruning_summary.pkl"
    with open(summary_file, 'wb') as f:
        pickle.dump({
            'original_dna': original_dna,
            'original_nonzero': np.count_nonzero(original_dna),
            'min_score_threshold': min_score,
            'tmax': tmax,
            'results_summary': results_summary,
            'timestamp': time.time()
        }, f)
    
    print(f"\n💾 Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Prune weights from DNA vectors while maintaining performance",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("dna_file", help="Path to pickled DNA file")
    parser.add_argument("--min-score", type=int, default=970,
                       help="Minimum acceptable total score (default: 970)")
    parser.add_argument("--tmax", type=int, default=5000,
                       help="Simulation time in ms (default: 5000)")
    parser.add_argument("--output-dir", default="pruning_results",
                       help="Output directory for results (default: pruning_results)")
    parser.add_argument("--method", choices=['greedy', 'impact', 'iterative', 'group2', 'group3', 'all'],
                       default='all', help="Pruning method to use (default: all)")
    
    args = parser.parse_args()
    
    if args.method == 'all':
        run_comprehensive_pruning(args.dna_file, args.min_score, args.tmax, args.output_dir)
    else:
        # Run single method
        data = pickle.load(open(args.dna_file, 'rb'))
        if isinstance(data, dict) and 'dna' in data:
            original_dna = data['dna']
        else:
            original_dna = data
        
        pruner = WeightPruner(original_dna, args.min_score, args.tmax)
        
        if args.method == 'greedy':
            result = pruner.greedy_magnitude_pruning()
        elif args.method == 'impact':
            result = pruner.impact_based_pruning()
        elif args.method == 'iterative':
            result = pruner.iterative_best_pruning()
        elif args.method == 'group2':
            result = pruner.group_removal_pruning(group_size=2)
        elif args.method == 'group3':
            result = pruner.group_removal_pruning(group_size=3)
        
        output_file = f"{args.method}_pruned_dna.pkl"
        pruner.save_results(result, args.method, output_file)


if __name__ == "__main__":
    main()