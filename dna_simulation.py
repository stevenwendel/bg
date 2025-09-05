#!/usr/bin/env python3
"""
DNA Simulation Module

Handles simulation of DNA vectors and voltage tracking.
"""

import numpy as np
from src.constants import (
    ACTIVE_SYNAPSES, NEURON_NAMES, TMAX, TONICALLY_ACTIVE_NEURONS, 
    BIN_SIZE
)
from src.validation import diagnose_conditions
from adaptive_tmax_fully_optimized import (
    initialize_connection_mapping, 
    get_cue_go_waves_for_tmax, 
    get_criteria_for_tmax,
    step_kernel_optimized,
    alpha, a, b, vreset, d, k, vr, vt, vpeak, C, E, N, ALPHA_L
)
from weight_pruning import evaluate_single_dna

def simulate_with_voltage_tracking(W, cue_wave, go_wave, alpha, tmax, control=False):
    """
    Run simulation with voltage tracking using the real adaptive_tmax simulation kernel.
    
    Returns:
        tuple of (voltage_history, spike_raster, score)
    """
    from numba import njit
    
    # Get criteria for this simulation
    crit_Exp, crit_Cont, crit_indices, pass_ids = get_criteria_for_tmax(tmax)
    
    @njit(fastmath=True, cache=True)
    def simulate_with_voltage_tracking_numba(W, a, b, vreset, d, k, vr, vt, vpeak, C, E, alpha, cue_wave, go_wave, 
                                           crit_Exp, crit_Cont, crit_indices, pass_ids, tmax, control):
        """Numba-compiled simulation with voltage tracking."""
        # Initialize state
        V = np.full(N, -60.0, np.float32)
        U = np.zeros_like(V)
        Ibuf = np.zeros((ALPHA_L, N), dtype=np.float32)
        HIST = np.zeros((N, BIN_SIZE), np.uint8)
        
        # Track voltage and spikes
        V_history = np.zeros((N, tmax), dtype=np.float32)
        spike_raster = np.zeros((N, tmax), dtype=np.uint8)
        
        score = 0
        t_ptr = 0
        bin = 0
        
        for t in range(tmax):
            # Apply stimuli (same as adaptive_tmax)
            if control == False:
                Ibuf[t_ptr, 0] += cue_wave[t]
            Ibuf[t_ptr, 7] += go_wave[t]
            
            # Run step kernel (same as adaptive_tmax)
            spk, t_ptr = step_kernel_optimized(V, U, Ibuf, t_ptr, a, b, vreset, d, k, vr, vt, vpeak, C, E, W, alpha)
            
            # Store voltage history with proper spike visualization
            V_history[:, t] = V.copy()
            
            # For visualization: show spike peak instead of reset voltage for spiking neurons
            for i in range(N):
                if spk[i] == 1:
                    V_history[i, t] = vpeak[i]  # Show the spike peak for visualization
            
            # Store spike raster
            spike_raster[:, t] = spk
            
            # Bin spikes for scoring (same as adaptive_tmax)
            cidx = t % BIN_SIZE
            HIST[:, cidx] = spk
            
            if cidx == (BIN_SIZE - 1):
                curr_bin_results = (np.sum(HIST, axis=1) >= 1).astype(np.uint8)
                crits = crit_Exp if (control == False) else crit_Cont
                bin_score = score_bin_optimized(curr_bin_results, crits, crit_indices, bin, pass_ids)
                score += bin_score
                bin += 1
        
        return V_history, spike_raster, score
    
    # Import the score function
    from adaptive_tmax_fully_optimized import score_bin_optimized
    
    # Call the numba-compiled simulation
    return simulate_with_voltage_tracking_numba(
        W, a, b, vreset, d, k, vr, vt, vpeak, C, E, alpha, cue_wave, go_wave,
        crit_Exp, crit_Cont, crit_indices, pass_ids, tmax, control
    )

def run_dna_with_voltage_tracking(dna_vector):
    """
    Run REAL simulation for a DNA vector using adaptive_tmax_fully_optimized and track actual voltage traces.
    
    Returns:
        dict with experimental and control results including REAL voltage traces and missed points
    """
    # Convert DNA to weight matrix
    conn_map = initialize_connection_mapping(ACTIVE_SYNAPSES, NEURON_NAMES)
    W = np.zeros((N, N), dtype=np.float32)
    for i, (pre_idx, post_idx) in enumerate(conn_map):
        W[pre_idx, post_idx] = float(dna_vector[i])
    
    # Get cue/go waves for simulation
    cue_wave, go_wave = get_cue_go_waves_for_tmax(TMAX)
    
    results = {'experimental': {}, 'control': {}}
    
    # Run both experimental and control conditions with REAL simulation
    for condition, control_flag in [('experimental', False), ('control', True)]:
        print(f"  Running {condition} condition...")
        
        try:
            # Run the REAL simulation with voltage tracking
            V_history, spike_raster, score = simulate_with_voltage_tracking(
                W, cue_wave, go_wave, alpha, TMAX, control=control_flag
            )
            
            # Convert voltage history to dictionary by neuron name
            voltages = {}
            for i, neuron_name in enumerate(NEURON_NAMES):
                voltages[neuron_name] = V_history[i, :]
            
            # Diagnose scoring misses using REAL spike raster
            missed_points = diagnose_conditions(spike_raster, condition, return_list=True)
            
            results[condition] = {
                'voltages': voltages,
                'missed_points': missed_points,
                'score': score
            }
            
            print(f"    ✅ {condition}: score={score}, missed={len(missed_points)} points")
            
        except Exception as e:
            print(f"    ❌ {condition} simulation failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to prevent total failure
            voltages = {}
            for neuron_name in NEURON_NAMES:
                voltages[neuron_name] = np.full(TMAX, -60.0, dtype=np.float32)
            
            results[condition] = {
                'voltages': voltages,
                'missed_points': [],
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