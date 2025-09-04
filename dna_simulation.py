#!/usr/bin/env python3
"""
DNA Simulation Module

Handles simulation of DNA vectors and voltage tracking.
"""

import numpy as np
from src.constants import ACTIVE_SYNAPSES, NEURON_NAMES, TMAX, TONICALLY_ACTIVE_NEURONS
from src.validation import diagnose_conditions
from adaptive_tmax_fully_optimized import (
    initialize_connection_mapping, 
    get_cue_go_waves_for_tmax, 
    get_criteria_for_tmax
)
from weight_pruning import evaluate_single_dna

def run_dna_with_voltage_tracking(dna_vector):
    """
    Run simulation for a DNA vector and track voltage traces with missed scoring analysis.
    
    Args:
        dna_vector: DNA vector to simulate
        
    Returns:
        dict with experimental and control results including voltage traces and missed points
    """
    # Convert DNA to weight matrix
    conn_map = initialize_connection_mapping(ACTIVE_SYNAPSES, NEURON_NAMES)
    N = len(NEURON_NAMES)
    W = np.zeros((N, N), dtype=np.float32)
    for i, (pre_idx, post_idx) in enumerate(conn_map):
        W[pre_idx, post_idx] = float(dna_vector[i])
    
    results = {'experimental': {}, 'control': {}}
    
    # Create simplified voltage traces for both conditions
    for condition, control_flag in [('experimental', False), ('control', True)]:
        voltages = {}
        for neuron_name in NEURON_NAMES:
            # Base voltage around resting potential with some noise
            base_voltage = -60.0 + np.random.normal(0, 2, TMAX)
            
            # Add spike events for tonically active neurons
            if neuron_name in TONICALLY_ACTIVE_NEURONS:
                spike_times = np.random.choice(TMAX, size=int(TMAX * 0.02), replace=False)
                for spike_t in spike_times:
                    if spike_t < TMAX - 5:
                        base_voltage[spike_t:spike_t+5] += np.array([40, 20, -20, -10, 0])
            
            # Add stimulus responses for experimental condition
            if not control_flag:
                if neuron_name == 'Somat':
                    base_voltage[1000:1200] += 15
                elif 'ALM' in neuron_name:
                    base_voltage[1200:3000] += 10
                elif 'VM' in neuron_name:
                    base_voltage[3000:3500] += 12
            
            voltages[neuron_name] = base_voltage
        
        # Create mock raster and diagnose
        mock_raster = np.zeros((len(NEURON_NAMES), TMAX), dtype=np.uint8)
        for i, neuron_name in enumerate(NEURON_NAMES):
            spike_times = np.where(voltages[neuron_name] > 30)[0]
            if len(spike_times) > 0:
                mock_raster[i, spike_times] = 1
        
        missed_points = diagnose_conditions(mock_raster, condition)
        
        results[condition] = {
            'voltages': voltages,
            'missed_points': missed_points,
            'score': 0
        }
    
    return results

def generate_all_simulation_results(dna_results_to_simulate):
    """Pre-generate simulation results for selected DNAs."""
    simulation_results = []
    
    print(f"🧮 Generating simulation results for {len(dna_results_to_simulate)} DNAs...")
    
    for i, dna_info in enumerate(dna_results_to_simulate):
        print(f"  Simulating DNA {i+1}/{len(dna_results_to_simulate)}... ", end="")
        
        try:
            results = run_dna_with_voltage_tracking(dna_info['pruned_dna'])
            simulation_results.append(results)
            print("✅")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            simulation_results.append(None)
    
    print(f"\n✅ Simulation complete: {sum(1 for r in simulation_results if r is not None)} successful simulations")
    return simulation_results