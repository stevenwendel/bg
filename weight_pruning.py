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


class GenerationBasedPruner:
    """Generation-based pruning algorithm that systematically removes weights across generations."""
    
    def __init__(self, pruning_threshold: int = 940, max_generations: int = 40, 
                 max_generation_size: int = 10000):
        """
        Initialize the generation-based pruner.
        
        Args:
            pruning_threshold: Score threshold for successful pruning
            max_generations: Maximum number of generations to explore
            max_generation_size: Maximum vectors per generation to prevent explosion
        """
        self.pruning_threshold = pruning_threshold
        self.max_generations = max_generations
        self.max_generation_size = max_generation_size
        
    def _evaluate_candidates(self, candidates: List[np.ndarray]) -> List[Dict]:
        """Evaluate a list of candidate DNA vectors."""
        evaluated = []
        for candidate in candidates:
            exp_score, cont_score, total_score = evaluate_single_dna(candidate)
            evaluated.append({
                'dna': candidate,
                'exp_score': exp_score,
                'cont_score': cont_score,
                'total_score': total_score
            })
        return evaluated
        
    def _generate_weight_removal_candidates(self, current_dna: np.ndarray) -> List[np.ndarray]:
        """Generate all possible single-weight removal candidates from current DNA."""
        candidates = []
        nonzero_indices = np.where(current_dna != 0)[0]
        
        for weight_idx in nonzero_indices:
            candidate = current_dna.copy()
            candidate[weight_idx] = 0
            candidates.append(candidate)
            
        return candidates
    
    def _filter_by_score_strategy(self, evaluated_candidates: List[Dict], 
                                 parent_score: int, strategy: str) -> List[Dict]:
        """Filter candidates based on scoring strategy."""
        if strategy == "improvement_only":
            return [c for c in evaluated_candidates if c['total_score'] > parent_score]
        elif strategy == "equal_or_greater":
            return [c for c in evaluated_candidates if c['total_score'] >= parent_score]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _remove_duplicates(self, dna_vectors: List[np.ndarray]) -> List[np.ndarray]:
        """Remove duplicate DNA vectors."""
        unique_vectors = []
        seen = set()
        
        for vec in dna_vectors:
            vec_tuple = tuple(vec)
            if vec_tuple not in seen:
                seen.add(vec_tuple)
                unique_vectors.append(vec)
                
        return unique_vectors
    
    def _limit_generation_size(self, evaluated_candidates: List[Dict]) -> List[Dict]:
        """Limit generation size by keeping top scoring candidates."""
        if len(evaluated_candidates) <= self.max_generation_size:
            return evaluated_candidates
            
        # Sort by score descending and keep top candidates
        sorted_candidates = sorted(evaluated_candidates, 
                                 key=lambda x: x['total_score'], reverse=True)
        return sorted_candidates[:self.max_generation_size]
    
    def _prune_single_dna(self, dna_info: Dict, dna_id: int) -> tuple[Dict, List[Dict]]:
        """
        Prune a single DNA vector using generation-based approach.
        
        Args:
            dna_info: Dictionary with 'dna' and 'total_score' keys
            dna_id: ID for tracking purposes
            
        Returns:
            Tuple of (pruning_result, successful_vectors)
        """
        print(f"\n🧬 Pruning DNA {dna_id} (Score: {dna_info['total_score']})...")
        
        current_generation = [dna_info['dna'].copy()]
        generation_num = 0
        successful_vectors = []
        best_vector = dna_info['dna'].copy()
        best_score = dna_info['total_score']
        
        print(f"  🎯 Starting with {np.count_nonzero(dna_info['dna'])} weights, "
              f"score: {dna_info['total_score']}")
        
        while current_generation and generation_num < self.max_generations:
            generation_num += 1
            next_generation = []
            
            print(f"  🌱 Generation {generation_num}: Processing {len(current_generation)} vectors...")
            
            for current_dna in current_generation:
                current_score = evaluate_single_dna(current_dna)[2]  # Get total score
                
                # Generate candidates by removing each weight
                candidates = self._generate_weight_removal_candidates(current_dna)
                if not candidates:
                    continue
                
                # Evaluate all candidates
                evaluated_candidates = self._evaluate_candidates(candidates)
                
                # Phase 1: Try improvement-only strategy
                improvement_candidates = self._filter_by_score_strategy(
                    evaluated_candidates, current_score, "improvement_only"
                )
                
                if improvement_candidates:
                    next_generation.extend([c['dna'] for c in improvement_candidates])
                    print(f"    📈 IMPROVEMENT-ONLY: Found {len(improvement_candidates)} improvements")
                else:
                    # Phase 2: Fall back to equal-or-greater strategy
                    equal_candidates = self._filter_by_score_strategy(
                        evaluated_candidates, current_score, "equal_or_greater"
                    )
                    if equal_candidates:
                        next_generation.extend([c['dna'] for c in equal_candidates])
                        print(f"    ⚖️  EQUAL-OR-GREATER: Found {len(equal_candidates)} candidates")
                
                # Check for successful vectors (exceeding threshold)
                for candidate in evaluated_candidates:
                    if candidate['total_score'] >= self.pruning_threshold:
                        successful_vectors.append({
                            'dna': candidate['dna'].copy(),
                            'score': candidate['total_score'],
                            'exp_score': candidate['exp_score'],
                            'cont_score': candidate['cont_score'],
                            'nonzero_weights': np.count_nonzero(candidate['dna']),
                            'generation': generation_num,
                            'original_dna_id': dna_id
                        })
                    
                    # Track best vector overall
                    if candidate['total_score'] > best_score:
                        best_vector = candidate['dna'].copy()
                        best_score = candidate['total_score']
            
            # Remove duplicates and limit generation size
            if next_generation:
                next_generation = self._remove_duplicates(next_generation)
                
                # If generation is too large, evaluate and keep best
                if len(next_generation) > self.max_generation_size:
                    evaluated_gen = self._evaluate_candidates(next_generation)
                    limited_gen = self._limit_generation_size(evaluated_gen)
                    next_generation = [c['dna'] for c in limited_gen]
                    print(f"    🔍 Limited to {len(next_generation)} vectors")
            
            current_generation = next_generation
        
        # Select best result
        if successful_vectors:
            best_successful = max(successful_vectors, key=lambda x: x['score'])
            final_vector = best_successful['dna']
            final_score = best_successful['score']
        else:
            final_vector = best_vector
            final_score = best_score
        
        # Final evaluation
        final_exp, final_cont, final_total = evaluate_single_dna(final_vector)
        
        result = {
            'original_dna': dna_info,
            'pruned_dna': final_vector,
            'original_score': dna_info['total_score'],
            'pruned_score': final_total,
            'original_nonzero': np.count_nonzero(dna_info['dna']),
            'pruned_nonzero': np.count_nonzero(final_vector),
            'weights_removed': np.count_nonzero(dna_info['dna']) - np.count_nonzero(final_vector),
            'final_exp_score': final_exp,
            'final_cont_score': final_cont,
            'generations_explored': generation_num,
            'successful_vectors_found': len(successful_vectors),
            'id': dna_id
        }
        
        reduction_pct = (result['weights_removed'] / result['original_nonzero']) * 100
        score_change = final_total - dna_info['total_score']
        
        print(f"  ✅ Best result: {result['original_nonzero']} → {result['pruned_nonzero']} weights "
              f"({reduction_pct:.1f}% reduction), Score: {result['original_score']} → "
              f"{result['pruned_score']} (+{score_change})")
        
        return result, successful_vectors
    
    def prune_dna_vectors(self, high_scoring_dnas: List[Dict]) -> tuple[List[Dict], List[Dict]]:
        """
        Prune multiple DNA vectors using generation-based approach.
        
        Args:
            high_scoring_dnas: List of dictionaries with 'dna' and 'total_score' keys
            
        Returns:
            Tuple of (pruned_results, all_successful_vectors)
        """
        pruned_results = []
        all_successful_vectors = []
        
        print(f"🔧 Starting generation-based pruning for {len(high_scoring_dnas)} DNA vectors...")
        print(f"⚡ Strategy: IMPROVEMENT-ONLY first, fallback to EQUAL-OR-GREATER")
        print(f"🎯 Success threshold: {self.pruning_threshold}")
        
        for i, dna_info in enumerate(high_scoring_dnas):
            try:
                result, successful_vectors = self._prune_single_dna(dna_info, i + 1)
                pruned_results.append(result)
                
                # Track all successful vectors found during pruning
                all_successful_vectors.extend(successful_vectors)
                    
            except Exception as e:
                print(f"  ❌ Error pruning DNA {i+1}: {e}")
                continue
        
        self._print_summary(pruned_results, all_successful_vectors)
        return pruned_results, all_successful_vectors
    
    def _fast_greedy_prune_single_dna(self, dna_info: Dict, dna_id: int, score_tolerance: int = 1) -> tuple[Dict, List[Dict]]:
        """
        Fast greedy pruning: Remove smallest weights first, with score tolerance.
        
        Algorithm:
        1. Phase 1: Remove weights maintaining equal-or-better score (multi-pass)
        2. Phase 2: Remove weights allowing score decrease up to tolerance (multi-pass)
        
        Args:
            dna_info: Dictionary with 'dna' and 'total_score' keys
            dna_id: ID for tracking purposes
            score_tolerance: Allow score to decrease by this much in phase 2 (default: 1)
            
        Returns:
            Tuple of (pruning_result, successful_vectors)
        """
        print(f"\n⚡ Fast greedy pruning DNA {dna_id} (Score: {dna_info['total_score']}, "
              f"Tolerance: -{score_tolerance})...")
        
        current_dna = dna_info['dna'].copy()
        original_score = dna_info['total_score']
        current_score = original_score
        successful_vectors = []
        total_removed = 0
        
        print(f"  🎯 Starting with {np.count_nonzero(current_dna)} weights, score: {current_score}")
        
        # PHASE 1: Equal-or-better score maintenance
        print(f"\n  🔶 PHASE 1: Maintaining equal-or-better score...")
        phase1_removed = 0
        pass_number = 0
        
        while True:
            pass_number += 1
            removed_this_pass = 0
            
            # Get all non-zero weights sorted by absolute magnitude (smallest first)
            nonzero_indices = np.where(current_dna != 0)[0]
            if len(nonzero_indices) == 0:
                break
            
            sorted_indices = sorted(nonzero_indices, key=lambda i: abs(current_dna[i]))
            
            print(f"    🔄 Phase 1 Pass {pass_number}: Testing {len(sorted_indices)} weights...")
            
            # Try removing each weight from smallest to largest
            for weight_idx in sorted_indices:
                original_value = current_dna[weight_idx]
                
                # Try removing this weight
                test_dna = current_dna.copy()
                test_dna[weight_idx] = 0
                
                exp_score, cont_score, total_score = evaluate_single_dna(test_dna)
                
                # Keep if score stays equal or improves
                if total_score >= current_score:
                    current_dna[weight_idx] = 0
                    current_score = total_score
                    removed_this_pass += 1
                    total_removed += 1
                    phase1_removed += 1
                    
                    print(f"      ✅ Removed weight {weight_idx} (value: {original_value:3d}) -> "
                          f"Score: {total_score}, Non-zero: {np.count_nonzero(current_dna)}")
                    
                    # Track if exceeds pruning threshold
                    if total_score >= self.pruning_threshold:
                        successful_vectors.append({
                            'dna': current_dna.copy(),
                            'score': total_score,
                            'exp_score': exp_score,
                            'cont_score': cont_score,
                            'nonzero_weights': np.count_nonzero(current_dna),
                            'pass': pass_number,
                            'phase': 1,
                            'original_dna_id': dna_id
                        })
            
            print(f"    📊 Phase 1 Pass {pass_number}: Removed {removed_this_pass} weights")
            
            # Stop if no weights were removed in this pass
            if removed_this_pass == 0:
                print(f"    🛑 Phase 1 complete: No more equal-or-better removals possible")
                break
                
            # Safety limit
            if pass_number > 20:
                print(f"    ⚠️ Phase 1 stopping at pass {pass_number}")
                break
        
        # PHASE 2: Tolerance-based pruning (if tolerance > 0)
        phase2_removed = 0
        if score_tolerance > 0:
            print(f"\n  🔶 PHASE 2: Allowing score decrease up to {score_tolerance} points...")
            min_acceptable_score = original_score - score_tolerance
            phase2_pass_number = 0
            
            while True:
                phase2_pass_number += 1
                removed_this_pass = 0
                
                # Get all non-zero weights sorted by absolute magnitude (smallest first)
                nonzero_indices = np.where(current_dna != 0)[0]
                if len(nonzero_indices) == 0:
                    break
                
                sorted_indices = sorted(nonzero_indices, key=lambda i: abs(current_dna[i]))
                
                print(f"    🔄 Phase 2 Pass {phase2_pass_number}: Testing {len(sorted_indices)} weights...")
                
                # Try removing each weight from smallest to largest
                for weight_idx in sorted_indices:
                    original_value = current_dna[weight_idx]
                    
                    # Try removing this weight
                    test_dna = current_dna.copy()
                    test_dna[weight_idx] = 0
                    
                    exp_score, cont_score, total_score = evaluate_single_dna(test_dna)
                    
                    # Keep if score stays within tolerance
                    if total_score >= min_acceptable_score:
                        current_dna[weight_idx] = 0
                        current_score = total_score
                        removed_this_pass += 1
                        total_removed += 1
                        phase2_removed += 1
                        
                        score_change = total_score - original_score
                        print(f"      ✅ Removed weight {weight_idx} (value: {original_value:3d}) -> "
                              f"Score: {total_score} ({score_change:+d}), Non-zero: {np.count_nonzero(current_dna)}")
                        
                        # Track if exceeds pruning threshold
                        if total_score >= self.pruning_threshold:
                            successful_vectors.append({
                                'dna': current_dna.copy(),
                                'score': total_score,
                                'exp_score': exp_score,
                                'cont_score': cont_score,
                                'nonzero_weights': np.count_nonzero(current_dna),
                                'pass': phase2_pass_number,
                                'phase': 2,
                                'original_dna_id': dna_id
                            })
                
                print(f"    📊 Phase 2 Pass {phase2_pass_number}: Removed {removed_this_pass} weights")
                
                # Stop if no weights were removed in this pass
                if removed_this_pass == 0:
                    print(f"    🛑 Phase 2 complete: No more tolerance-based removals possible")
                    break
                    
                # Safety limit
                if phase2_pass_number > 20:
                    print(f"    ⚠️ Phase 2 stopping at pass {phase2_pass_number}")
                    break
        
        # Final evaluation
        final_exp, final_cont, final_total = evaluate_single_dna(current_dna)
        
        result = {
            'original_dna': dna_info,
            'pruned_dna': current_dna,
            'original_score': dna_info['total_score'],
            'pruned_score': final_total,
            'original_nonzero': np.count_nonzero(dna_info['dna']),
            'pruned_nonzero': np.count_nonzero(current_dna),
            'weights_removed': total_removed,
            'phase1_removed': phase1_removed,
            'phase2_removed': phase2_removed,
            'final_exp_score': final_exp,
            'final_cont_score': final_cont,
            'phase1_passes': pass_number,
            'phase2_passes': phase2_pass_number if score_tolerance > 0 else 0,
            'successful_vectors_found': len(successful_vectors),
            'score_tolerance_used': score_tolerance,
            'id': dna_id
        }
        
        reduction_pct = (result['weights_removed'] / result['original_nonzero']) * 100
        score_change = final_total - dna_info['total_score']
        
        print(f"  ✅ Fast greedy result: {result['original_nonzero']} → {result['pruned_nonzero']} weights "
              f"({reduction_pct:.1f}% reduction)")
        print(f"    Score: {result['original_score']} → {result['pruned_score']} ({score_change:+d})")
        print(f"    Phase 1: {phase1_removed} weights | Phase 2: {phase2_removed} weights")
        
        return result, successful_vectors
    
    def fast_greedy_prune_dna_vectors(self, high_scoring_dnas: List[Dict], score_tolerance: int = 1) -> tuple[List[Dict], List[Dict]]:
        """
        Fast greedy pruning for multiple DNA vectors with score tolerance.
        
        Args:
            high_scoring_dnas: List of DNA info dictionaries
            score_tolerance: Allow score to decrease by this much in phase 2 (default: 1)
            
        Returns:
            Tuple of (pruned_results, all_successful_vectors)
        """
        pruned_results = []
        all_successful_vectors = []
        
        print(f"⚡ Starting FAST GREEDY pruning for {len(high_scoring_dnas)} DNA vectors...")
        print(f"🎯 Strategy: Phase 1 (equal-or-better) + Phase 2 (tolerance: -{score_tolerance})")
        print(f"🎯 Success threshold: {self.pruning_threshold}")
        
        for i, dna_info in enumerate(high_scoring_dnas):
            try:
                result, successful_vectors = self._fast_greedy_prune_single_dna(dna_info, i + 1, score_tolerance)
                pruned_results.append(result)
                
                # Track all successful vectors found during pruning
                all_successful_vectors.extend(successful_vectors)
                    
            except Exception as e:
                print(f"  ❌ Error pruning DNA {i+1}: {e}")
                continue
        
        self._print_fast_summary(pruned_results, all_successful_vectors, score_tolerance)
        return pruned_results, all_successful_vectors
    
    def _print_fast_summary(self, pruned_results: List[Dict], successful_vectors: List[Dict], score_tolerance: int = 1):
        """Print summary specifically for fast greedy pruning."""
        print(f"\n✅ Fast greedy pruning complete: {len(pruned_results)} DNA vectors processed")
        print(f"🎯 SUCCESSFUL VECTORS: {len(successful_vectors)} exceeded threshold "
              f"({self.pruning_threshold})")
        
        if not pruned_results:
            return
        
        # Summary statistics
        avg_reduction = np.mean([
            r['weights_removed']/r['original_nonzero']*100 for r in pruned_results
        ])
        total_original = sum(r['original_nonzero'] for r in pruned_results)
        total_pruned = sum(r['pruned_nonzero'] for r in pruned_results)
        
        # Phase statistics
        avg_phase1 = np.mean([r.get('phase1_removed', 0) for r in pruned_results])
        avg_phase2 = np.mean([r.get('phase2_removed', 0) for r in pruned_results])
        avg_phase1_passes = np.mean([r.get('phase1_passes', 0) for r in pruned_results])
        avg_phase2_passes = np.mean([r.get('phase2_passes', 0) for r in pruned_results])
        
        print(f"\n📊 Fast Greedy Pruning Summary:")
        print(f"  Score tolerance used: -{score_tolerance}")
        print(f"  Average weight reduction: {avg_reduction:.1f}%")
        print(f"  Total weights: {total_original} → {total_pruned}")
        print(f"  Best pruned score: {max(r['pruned_score'] for r in pruned_results)}")
        print(f"  Most efficient (fewest weights): {min(r['pruned_nonzero'] for r in pruned_results)} weights")
        print(f"  Phase 1 average: {avg_phase1:.1f} weights in {avg_phase1_passes:.1f} passes")
        print(f"  Phase 2 average: {avg_phase2:.1f} weights in {avg_phase2_passes:.1f} passes")
        
        # Score improvement analysis
        score_improvements = [r['pruned_score'] - r['original_score'] for r in pruned_results]
        print(f"\n📈 Score changes:")
        print(f"  Average change: {np.mean(score_improvements):+.1f}")
        print(f"  Best change: {max(score_improvements):+d}")
        print(f"  Worst change: {min(score_improvements):+d}")
        
        if successful_vectors:
            print(f"\n🎯 Successful vectors (>= {self.pruning_threshold}):")
            for sv in successful_vectors:
                phase_text = f"Phase {sv['phase']}" if 'phase' in sv else f"Pass {sv.get('pass', '?')}"
                print(f"  DNA {sv['original_dna_id']}: {sv['score']} points, "
                      f"{sv['nonzero_weights']} weights, {phase_text}")

    def _print_summary(self, pruned_results: List[Dict], successful_vectors: List[Dict]):
        """Print comprehensive summary of pruning results."""
        print(f"\n✅ Generation-based pruning complete: {len(pruned_results)} DNA vectors processed")
        print(f"🎯 SUCCESSFUL VECTORS: {len(successful_vectors)} exceeded threshold "
              f"({self.pruning_threshold})")
        
        if not pruned_results:
            return
        
        # Summary statistics
        avg_reduction = np.mean([
            r['weights_removed']/r['original_nonzero']*100 for r in pruned_results
        ])
        total_original = sum(r['original_nonzero'] for r in pruned_results)
        total_pruned = sum(r['pruned_nonzero'] for r in pruned_results)
        
        print(f"\n📊 Generation-Based Pruning Summary:")
        print(f"  Average weight reduction: {avg_reduction:.1f}%")
        print(f"  Total weights: {total_original} → {total_pruned}")
        print(f"  Best pruned score: {max(r['pruned_score'] for r in pruned_results)}")
        print(f"  Most efficient (fewest weights): {min(r['pruned_nonzero'] for r in pruned_results)} weights")
        print(f"  Average generations explored: {np.mean([r['generations_explored'] for r in pruned_results]):.1f}")
        
        # Score improvement analysis
        score_improvements = [r['pruned_score'] - r['original_score'] for r in pruned_results]
        print(f"\n📈 Score improvements:")
        print(f"  Average improvement: +{np.mean(score_improvements):.1f}")
        print(f"  Best improvement: +{max(score_improvements)}")
        print(f"  Worst change: +{min(score_improvements)}")
        
        if successful_vectors:
            print(f"\n🎯 Successful vectors (>= {self.pruning_threshold}):")
            for sv in successful_vectors:
                print(f"  DNA {sv['original_dna_id']}: {sv['score']} points, "
                      f"{sv['nonzero_weights']} weights, gen {sv['generation']}")


def evaluate_single_dna_fast(dna_vector: np.ndarray) -> Tuple[int, int, int]:
    """Fast evaluation using existing evaluate_single_dna function."""
    return evaluate_single_dna(dna_vector, 5000)


def prune_dna_vectors(high_scoring_dnas: List[Dict], pruning_threshold: int = 940, 
                     max_generations: int = 40, max_generation_size: int = 10000,
                     method: str = "generation_based", score_tolerance: int = 1) -> tuple[List[Dict], List[Dict]]:
    """
    Convenience function for DNA pruning with multiple algorithm options.
    
    Args:
        high_scoring_dnas: List of DNA info dictionaries with 'dna' and 'total_score' keys
        pruning_threshold: Score threshold for considering pruning successful
        max_generations: Maximum generations to explore per DNA (generation_based only)
        max_generation_size: Maximum vectors per generation (generation_based only)
        method: Pruning method to use ("generation_based" or "fast_greedy")
        score_tolerance: Allow score to decrease by this much (fast_greedy only, default: 1)
        
    Returns:
        Tuple of (pruned_results, successful_vectors_during_pruning)
    """
    pruner = GenerationBasedPruner(
        pruning_threshold=pruning_threshold,
        max_generations=max_generations, 
        max_generation_size=max_generation_size
    )
    
    if method == "fast_greedy":
        return pruner.fast_greedy_prune_dna_vectors(high_scoring_dnas, score_tolerance)
    elif method == "generation_based":
        return pruner.prune_dna_vectors(high_scoring_dnas)
    else:
        raise ValueError(f"Unknown pruning method: {method}. Use 'fast_greedy' or 'generation_based'")


if __name__ == "__main__":
    main()