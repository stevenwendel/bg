#!/usr/bin/env python3
"""
Adaptive GA Parameter Optimizer using Bayesian Optimization

This module provides Bayesian optimization for GA hyperparameters across multiple runs,
automatically finding optimal configurations that produce the highest scoring results.
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from copy import deepcopy

# Try to import scikit-learn for Gaussian Process (if available)
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class SimpleGaussianProcess:
    """Simple Gaussian Process implementation for when sklearn is not available."""
    
    def __init__(self, kernel_scale=1.0, noise_level=0.1):
        self.kernel_scale = kernel_scale
        self.noise_level = noise_level
        self.X_train = None
        self.y_train = None
        self.scaler_X = None
        self.scaler_y = None
    
    def fit(self, X, y):
        """Fit the GP to training data."""
        X = np.array(X)
        y = np.array(y)
        
        # Scale inputs and outputs
        self.scaler_X = StandardScaler() if SKLEARN_AVAILABLE else None
        self.scaler_y = StandardScaler() if SKLEARN_AVAILABLE else None
        
        if self.scaler_X:
            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        else:
            # Simple scaling without sklearn
            X_scaled = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
            y_scaled = (y - np.mean(y)) / (np.std(y) + 1e-8)
        
        self.X_train = X_scaled
        self.y_train = y_scaled
    
    def predict(self, X_test):
        """Make predictions with uncertainty estimates."""
        if self.X_train is None:
            # No training data yet, return random predictions
            n_test = len(X_test)
            mean = np.zeros(n_test)
            std = np.ones(n_test)
            return mean, std
        
        X_test = np.array(X_test)
        
        # Scale test inputs
        if self.scaler_X:
            X_test_scaled = self.scaler_X.transform(X_test)
        else:
            X_test_scaled = (X_test - np.mean(self.X_train, axis=0)) / (np.std(self.X_train, axis=0) + 1e-8)
        
        # Simple kernel-based prediction
        n_test = X_test_scaled.shape[0]
        mean = np.zeros(n_test)
        std = np.ones(n_test) * 0.5  # Default uncertainty
        
        for i in range(n_test):
            # Compute kernel similarities
            if len(self.X_train) > 0:
                diffs = self.X_train - X_test_scaled[i]
                distances = np.sum(diffs**2, axis=1)
                similarities = np.exp(-distances / (2 * self.kernel_scale**2))
                
                # Weighted average prediction
                if np.sum(similarities) > 1e-8:
                    mean[i] = np.sum(similarities * self.y_train) / np.sum(similarities)
                    # Uncertainty decreases with similarity to training points
                    std[i] = max(0.1, 1.0 - np.max(similarities))
        
        # Scale back to original space
        if self.scaler_y:
            mean = self.scaler_y.inverse_transform(mean.reshape(-1, 1)).flatten()
            std = std * self.scaler_y.scale_
        else:
            mean = mean * (np.std(self.y_train) + 1e-8) + np.mean(self.y_train)
        
        return mean, std

class AdaptiveGAOptimizer:
    """
    Bayesian optimization for GA hyperparameters.
    
    Optimizes MUT_RATE, MUT_SIGMA, ELITE_SIZE, RANK_DEPTH, NUM_GENERATIONS, POP_SIZE
    while maintaining constraints on total computational budget.
    """
    
    def __init__(self, base_config: Dict, total_simulation_budget: int = None,
                 budget_tolerance: float = 0.05):
        """
        Initialize the optimizer.
        
        Args:
            base_config: Initial GA configuration from constants
            total_simulation_budget: Total NUM_GENERATIONS * POP_SIZE budget
            budget_tolerance: Allowable deviation from budget (0.05 = 5%)
        """
        self.base_config = deepcopy(base_config)
        
        # Calculate total simulation budget
        if total_simulation_budget is None:
            self.total_simulation_budget = base_config["NUM_GENERATIONS"] * base_config["POP_SIZE"]
        else:
            self.total_simulation_budget = total_simulation_budget
        
        self.budget_tolerance = budget_tolerance
        self.budget_min = int(self.total_simulation_budget * (1 - budget_tolerance))
        self.budget_max = int(self.total_simulation_budget * (1 + budget_tolerance))
        
        # Define parameter bounds (will be normalized to [0,1])
        self.param_bounds = {
            'MUT_RATE': (0.1, 0.9),         # Mutation rate
            'MUT_SIGMA': (0.1, 3.0),        # Mutation sigma
            'ELITE_SIZE': (5, 50),          # Elite size
            'RANK_DEPTH': (50, 1000),       # Rank depth
            'NUM_GENERATIONS': (50, 1000),   # Generations
            'POP_SIZE': (100, 2000)         # Population size
        }
        
        # Initialize GP model
        if SKLEARN_AVAILABLE:
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
            self.gp_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=10,
                random_state=42
            )
        else:
            print("⚠️ sklearn not available, using simple GP implementation")
            self.gp_model = SimpleGaussianProcess()
        
        # History tracking
        self.optimization_history = []
        self.parameter_history = []
        self.score_history = []
        
        # Best configuration tracking
        self.best_config = None
        self.best_score = -np.inf
        
    def normalize_params(self, params: Dict) -> np.ndarray:
        """Normalize parameters to [0,1] range for GP."""
        normalized = []
        param_names = ['MUT_RATE', 'MUT_SIGMA', 'ELITE_SIZE', 'RANK_DEPTH', 'NUM_GENERATIONS', 'POP_SIZE']
        
        for name in param_names:
            value = params[name]
            min_val, max_val = self.param_bounds[name]
            normalized_val = (value - min_val) / (max_val - min_val)
            normalized.append(np.clip(normalized_val, 0, 1))
        
        return np.array(normalized)
    
    def denormalize_params(self, normalized_params: np.ndarray) -> Dict:
        """Convert normalized parameters back to original scale."""
        param_names = ['MUT_RATE', 'MUT_SIGMA', 'ELITE_SIZE', 'RANK_DEPTH', 'NUM_GENERATIONS', 'POP_SIZE']
        params = {}
        
        for i, name in enumerate(param_names):
            min_val, max_val = self.param_bounds[name]
            denorm_val = min_val + normalized_params[i] * (max_val - min_val)
            
            # Round integer parameters
            if name in ['ELITE_SIZE', 'RANK_DEPTH', 'NUM_GENERATIONS', 'POP_SIZE']:
                params[name] = int(round(denorm_val))
            else:
                params[name] = denorm_val
        
        return params
    
    def apply_constraints(self, params: Dict) -> Dict:
        """Apply constraints to parameters (budget, validity checks)."""
        params = deepcopy(params)
        
        # Ensure budget constraint
        current_budget = params['NUM_GENERATIONS'] * params['POP_SIZE']
        
        if current_budget < self.budget_min or current_budget > self.budget_max:
            # Adjust to maintain budget while preferring the parameter that's more important
            target_budget = self.total_simulation_budget
            
            # If generations is too high/low, adjust population first
            if current_budget > self.budget_max:
                # Reduce population to fit budget
                params['POP_SIZE'] = max(self.param_bounds['POP_SIZE'][0], 
                                       int(target_budget / params['NUM_GENERATIONS']))
            elif current_budget < self.budget_min:
                # Increase population to fit budget  
                params['POP_SIZE'] = min(self.param_bounds['POP_SIZE'][1],
                                       int(target_budget / params['NUM_GENERATIONS']))
        
        # Apply parameter bounds
        for name, (min_val, max_val) in self.param_bounds.items():
            if name in ['ELITE_SIZE', 'RANK_DEPTH', 'NUM_GENERATIONS', 'POP_SIZE']:
                params[name] = int(np.clip(params[name], min_val, max_val))
            else:
                params[name] = np.clip(params[name], min_val, max_val)
        
        # Logical constraints
        params['ELITE_SIZE'] = min(params['ELITE_SIZE'], params['POP_SIZE'] // 10)  # Elite ≤ 10% of population
        params['RANK_DEPTH'] = min(params['RANK_DEPTH'], params['POP_SIZE'])       # Rank depth ≤ population
        
        return params
    
    def acquisition_function(self, X: np.ndarray, exploitation_weight: float = 0.1) -> np.ndarray:
        """
        Upper Confidence Bound acquisition function.
        
        Args:
            X: Normalized parameter vectors
            exploitation_weight: Balance between exploration and exploitation
        """
        if len(self.score_history) == 0:
            # No data yet, return random scores to encourage exploration
            return np.random.rand(len(X))
        
        # Get predictions from GP
        if SKLEARN_AVAILABLE and hasattr(self.gp_model, 'predict'):
            mean, std = self.gp_model.predict(X, return_std=True)
        else:
            mean, std = self.gp_model.predict(X)
        
        # Upper confidence bound
        ucb = mean + (1 - exploitation_weight) * std
        
        return ucb
    
    def suggest_parameters(self, run_number: int, total_runs: int) -> Dict:
        """
        Suggest the next set of parameters using Bayesian optimization.
        
        Args:
            run_number: Current run number (1-indexed)
            total_runs: Total number of runs planned
        """
        if run_number == 1:
            # First run: use base configuration
            config = deepcopy(self.base_config)
            print(f"\n🔧 Run {run_number} Configuration (BASELINE):")
        else:
            # Subsequent runs: optimize using GP
            
            # Fit GP model if we have data
            if len(self.parameter_history) > 0:
                X_train = np.array(self.parameter_history)
                y_train = np.array(self.score_history)
                
                self.gp_model.fit(X_train, y_train)
            
            # Generate candidate configurations
            n_candidates = 100
            exploitation_weight = run_number / total_runs  # Increase exploitation over time
            
            # Random candidates
            candidates = np.random.rand(n_candidates, 6)
            
            # Add some systematic exploration around best known configuration
            if self.best_config is not None:
                best_normalized = self.normalize_params(self.best_config)
                # Add perturbations around best configuration
                n_local = 20
                local_candidates = np.random.normal(
                    best_normalized, 
                    scale=0.1, 
                    size=(n_local, 6)
                )
                local_candidates = np.clip(local_candidates, 0, 1)
                candidates = np.vstack([candidates, local_candidates])
            
            # Evaluate acquisition function
            acquisition_scores = self.acquisition_function(candidates, exploitation_weight)
            
            # Select best candidate
            best_idx = np.argmax(acquisition_scores)
            best_candidate = candidates[best_idx]
            
            # Convert back to parameter space and apply constraints
            params = self.denormalize_params(best_candidate)
            config = deepcopy(self.base_config)
            config.update(params)
            config = self.apply_constraints(config)
            
            print(f"\n🔧 Run {run_number} Configuration (OPTIMIZED):")
            print(f"   Exploitation weight: {exploitation_weight:.2f}")
            print(f"   Expected improvement: {acquisition_scores[best_idx]:.3f}")
        
        # Calculate and display budget
        current_budget = config['NUM_GENERATIONS'] * config['POP_SIZE']
        budget_deviation = (current_budget - self.total_simulation_budget) / self.total_simulation_budget * 100
        
        print(f"   Budget: {current_budget:,} simulations ({budget_deviation:+.1f}% from baseline)")
        
        # Display configuration changes
        for param_name in ['MUT_RATE', 'MUT_SIGMA', 'ELITE_SIZE', 'RANK_DEPTH', 'NUM_GENERATIONS', 'POP_SIZE']:
            old_val = self.base_config[param_name]
            new_val = config[param_name]
            
            if param_name in ['MUT_RATE', 'MUT_SIGMA']:
                change_str = f"{old_val:.3f} → {new_val:.3f}"
            else:
                change_str = f"{old_val} → {new_val}"
            
            if new_val != old_val:
                change_pct = (new_val - old_val) / old_val * 100
                print(f"   {param_name:15s}: {change_str} ({change_pct:+.1f}%)")
            else:
                print(f"   {param_name:15s}: {change_str}")
        
        return config
    
    def record_result(self, config: Dict, best_score: float, run_duration: float, 
                     additional_metrics: Dict = None):
        """
        Record the result of a GA run for optimization.
        
        Args:
            config: Configuration used for this run
            best_score: Best score achieved in this run
            run_duration: How long the run took
            additional_metrics: Additional metrics (total_individuals, etc.)
        """
        # Normalize parameters for GP
        normalized_params = self.normalize_params(config)
        
        # Store history
        self.parameter_history.append(normalized_params)
        self.score_history.append(best_score)
        
        # Update best configuration
        if best_score > self.best_score:
            self.best_score = best_score
            self.best_config = deepcopy(config)
        
        # Store detailed history
        record = {
            'run_number': len(self.optimization_history) + 1,
            'timestamp': datetime.now().isoformat(),
            'config': deepcopy(config),
            'best_score': best_score,
            'run_duration': run_duration,
            'normalized_params': normalized_params.tolist(),
            'is_best_so_far': best_score >= self.best_score,
            'improvement_over_baseline': best_score - (self.score_history[0] if self.score_history else 0)
        }
        
        if additional_metrics:
            record['metrics'] = additional_metrics
        
        self.optimization_history.append(record)
        
        print(f"\n📊 Run {len(self.optimization_history)} Results:")
        print(f"   Best score: {best_score:.1f}")
        print(f"   Duration: {run_duration:.1f}s")
        print(f"   Best so far: {self.best_score:.1f} {'🎯' if best_score >= self.best_score else ''}")
        
        if len(self.score_history) > 1:
            improvement = best_score - self.score_history[0]
            print(f"   Improvement over baseline: {improvement:+.1f}")
    
    def get_optimization_summary(self) -> Dict:
        """Get a summary of the optimization process."""
        if not self.optimization_history:
            return {}
        
        scores = [r['best_score'] for r in self.optimization_history]
        durations = [r['run_duration'] for r in self.optimization_history]
        
        summary = {
            'total_runs': len(self.optimization_history),
            'baseline_score': scores[0] if scores else 0,
            'best_score_achieved': max(scores) if scores else 0,
            'final_score': scores[-1] if scores else 0,
            'total_improvement': max(scores) - scores[0] if len(scores) > 1 else 0,
            'average_score': np.mean(scores) if scores else 0,
            'score_std': np.std(scores) if len(scores) > 1 else 0,
            'best_config': deepcopy(self.best_config) if self.best_config else None,
            'optimization_efficiency': (max(scores) - scores[0]) / len(scores) if len(scores) > 1 else 0,
            'total_duration': sum(durations) if durations else 0,
            'average_duration': np.mean(durations) if durations else 0
        }
        
        return summary
    
    def save_optimization_history(self, filepath: str):
        """Save optimization history to file."""
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        data = {
            'base_config': convert_numpy(self.base_config),
            'total_simulation_budget': int(self.total_simulation_budget),
            'budget_tolerance': float(self.budget_tolerance),
            'param_bounds': convert_numpy(self.param_bounds),
            'optimization_history': convert_numpy(self.optimization_history),
            'best_config': convert_numpy(self.best_config),
            'best_score': float(self.best_score) if self.best_score != -np.inf else None,
            'summary': convert_numpy(self.get_optimization_summary()),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📁 Optimization history saved to: {filepath}")
    
    def load_optimization_history(self, filepath: str):
        """Load optimization history from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.base_config = data['base_config']
        self.total_simulation_budget = data['total_simulation_budget']
        self.budget_tolerance = data['budget_tolerance']
        self.param_bounds = data['param_bounds']
        self.optimization_history = data['optimization_history']
        self.best_config = data['best_config']
        self.best_score = data['best_score']
        
        # Rebuild parameter and score histories
        self.parameter_history = [np.array(r['normalized_params']) for r in self.optimization_history]
        self.score_history = [r['best_score'] for r in self.optimization_history]
        
        print(f"📁 Optimization history loaded from: {filepath}")
        print(f"   Loaded {len(self.optimization_history)} runs")
        print(f"   Best score so far: {self.best_score:.1f}")

def print_optimization_summary(optimizer: AdaptiveGAOptimizer):
    """Print a detailed optimization summary."""
    summary = optimizer.get_optimization_summary()
    
    if not summary:
        print("No optimization data available")
        return
    
    print("\n" + "=" * 60)
    print("🎯 BAYESIAN OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"Total runs: {summary['total_runs']}")
    print(f"Baseline score: {summary['baseline_score']:.1f}")
    print(f"Best score achieved: {summary['best_score_achieved']:.1f}")
    print(f"Final score: {summary['final_score']:.1f}")
    print(f"Total improvement: {summary['total_improvement']:+.1f}")
    print(f"Optimization efficiency: {summary['optimization_efficiency']:.2f} points/run")
    print(f"Average score: {summary['average_score']:.1f} ± {summary['score_std']:.1f}")
    
    if summary['best_config']:
        print(f"\n🏆 Best Configuration Found:")
        base_config = optimizer.base_config
        best_config = summary['best_config']
        
        for param in ['MUT_RATE', 'MUT_SIGMA', 'ELITE_SIZE', 'RANK_DEPTH', 'NUM_GENERATIONS', 'POP_SIZE']:
            base_val = base_config[param]
            best_val = best_config[param]
            
            if param in ['MUT_RATE', 'MUT_SIGMA']:
                change_str = f"{base_val:.3f} → {best_val:.3f}"
            else:
                change_str = f"{base_val} → {best_val}"
            
            if best_val != base_val:
                change_pct = (best_val - base_val) / base_val * 100
                print(f"   {param:15s}: {change_str} ({change_pct:+.1f}%)")
            else:
                print(f"   {param:15s}: {change_str}")
        
        best_budget = best_config['NUM_GENERATIONS'] * best_config['POP_SIZE']
        base_budget = base_config['NUM_GENERATIONS'] * base_config['POP_SIZE']
        budget_change = (best_budget - base_budget) / base_budget * 100
        print(f"   {'BUDGET':15s}: {base_budget:,} → {best_budget:,} ({budget_change:+.1f}%)")
    
    print("=" * 60)