#!/usr/bin/env python3
"""
Test script for DNA browser functionality using new_jh_weights from constants.py

This script demonstrates the enhanced DNA browser with missed scoring visualization
by creating mock data that resembles the actual GA analysis results structure.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append('src')
sys.path.append('.')

from src.constants import (
    new_jh_weights, ACTIVE_SYNAPSES, NEURON_NAMES, TMAX, BIN_SIZE,
    CRITERIA_NAMES, INHIBITORY_NEURONS
)
from src.validation import diagnose_conditions
from dna_browser import create_dual_dna_browser
from dna_visualization import create_voltage_plot
from dna_simulation import simulate_with_voltage_tracking
import matplotlib.pyplot as plt

def convert_weights_to_dna_vector(weight_connections, active_synapses):
    """Convert connection weights list to DNA vector format."""
    # Create a dictionary for fast lookup
    weight_dict = {(pre, post): weight for pre, post, weight in weight_connections}
    
    # Create DNA vector matching ACTIVE_SYNAPSES order
    dna_vector = []
    for pre, post in active_synapses:
        weight = weight_dict.get((pre, post), 0)
        # Ensure positive values for DNA encoding (negative weights handled by neuron types)
        dna_vector.append(abs(weight))
    
    return np.array(dna_vector, dtype=np.int32)

def create_mock_voltage_and_spike_data(dna_vector, condition="experimental"):
    """Create realistic mock voltage traces and spike raster data."""
    # Simulate voltage traces for each neuron
    voltages = {}
    spike_raster = np.zeros((len(NEURON_NAMES), TMAX), dtype=np.uint8)
    
    for i, neuron_name in enumerate(NEURON_NAMES):
        # Create realistic voltage trace with some variability
        base_voltage = -60.0
        noise = np.random.normal(0, 2, TMAX)
        
        # Add some stimulus-related activity
        stimulus_times = []
        if condition == "experimental":
            if neuron_name == "Somat":
                # Somat should be active during stimulus (1000-2000ms)
                stimulus_times = list(range(1000, 2000, 200))
            elif neuron_name in ["ALMprep", "ALMinter"]:
                # ALM neurons active during preparation
                stimulus_times = list(range(1200, 3000, 300))
            elif neuron_name == "ALMresp":
                # ALMresp active during response
                stimulus_times = list(range(3000, 4000, 250))
            elif neuron_name in ["VMprep", "VMresp"]:
                # VM neurons show sustained activity
                stimulus_times = list(range(1500, 4000, 400))
        
        # Generate voltage trace
        voltage = base_voltage + noise
        
        # Add spikes at stimulus times
        for spike_time in stimulus_times:
            if spike_time < TMAX:
                # Add spike waveform
                spike_start = max(0, spike_time - 5)
                spike_end = min(TMAX, spike_time + 10)
                voltage[spike_start:spike_end] += np.linspace(0, 35, spike_end - spike_start)
                spike_raster[i, spike_time] = 1
        
        voltages[neuron_name] = voltage
    
    return voltages, spike_raster

def create_mock_dna_info(dna_id, dna_vector, original_score=975, pruned_score=980):
    """Create mock DNA info structure matching the expected format."""
    nonzero_weights = np.count_nonzero(dna_vector)
    
    return {
        'id': dna_id,
        'original_dna': {
            'run_folder': 'test_new_jh_weights',
            'generation': 42,
            'process_id': 1,
            'original_dna_id': f'test_dna_{dna_id}'
        },
        'pruned_dna': dna_vector,
        'original_score': original_score,
        'pruned_score': pruned_score,
        'original_nonzero': nonzero_weights + 5,  # Simulate some pruning
        'pruned_nonzero': nonzero_weights,
        'weights_removed': 5,
        'final_exp_score': int(pruned_score * 0.49),  # Approximate split
        'final_cont_score': int(pruned_score * 0.51),
    }

def create_mock_simulation_results(dna_vector):
    """Create mock simulation results with both experimental and control conditions."""
    # Generate voltage and spike data for both conditions
    exp_voltages, exp_spikes = create_mock_voltage_and_spike_data(dna_vector, "experimental")
    cont_voltages, cont_spikes = create_mock_voltage_and_spike_data(dna_vector, "control")
    
    # Calculate mock scores using actual diagnose_conditions
    exp_missed = diagnose_conditions(exp_spikes, "experimental", return_list=True)
    cont_missed = diagnose_conditions(cont_spikes, "control", return_list=True)
    
    max_score_per_condition = TMAX // BIN_SIZE * len(CRITERIA_NAMES)
    exp_score = max_score_per_condition - len(exp_missed)
    cont_score = max_score_per_condition - len(cont_missed)
    
    return {
        'experimental': {
            'voltages': exp_voltages,
            'spike_raster': exp_spikes,
            'score': exp_score,
            'missed_diagnoses': exp_missed
        },
        'control': {
            'voltages': cont_voltages,
            'spike_raster': cont_spikes,
            'score': cont_score,
            'missed_diagnoses': cont_missed
        }
    }

def test_dna_browser_with_new_jh_weights():
    """
    Test the DNA browser functionality using the new_jh_weights DNA vector.
    
    This function creates mock data structures that match the format expected
    by the DNA browser and demonstrates all the enhanced features including
    missed scoring visualization.
    """
    print("🧪 Testing DNA Browser with new_jh_weights")
    print("=" * 50)
    
    # Convert new_jh_weights to DNA vector format
    print("📊 Converting new_jh_weights to DNA vector format...")
    dna_vector = convert_weights_to_dna_vector(new_jh_weights, ACTIVE_SYNAPSES)
    
    print(f"✅ DNA vector created with {len(dna_vector)} connections")
    print(f"   Non-zero weights: {np.count_nonzero(dna_vector)}")
    print(f"   Weight range: {dna_vector.min()} - {dna_vector.max()}")
    
    # Create mock DNA info
    print("\n🧬 Creating mock DNA analysis results...")
    target_dnas = [
        create_mock_dna_info(1, dna_vector, original_score=970, pruned_score=980)
    ]
    
    # Create mock simulation results
    print("🔬 Generating mock simulation data...")
    simulation_results = [
        create_mock_simulation_results(dna_vector)
    ]
    
    # Print connection details
    print(f"\n🔗 new_jh_weights connections ({len(new_jh_weights)} total):")
    for i, (pre, post, weight) in enumerate(new_jh_weights):
        is_inhibitory = pre in INHIBITORY_NEURONS
        conn_type = "Inhibitory" if is_inhibitory else "Excitatory"
        print(f"  {i+1:2d}. {pre:8s} → {post:8s} | {weight:6d} | {conn_type}")
    
    # Show missed scoring analysis
    exp_missed = simulation_results[0]['experimental']['missed_diagnoses']
    cont_missed = simulation_results[0]['control']['missed_diagnoses']
    
    print(f"\n📉 Mock Scoring Analysis:")
    print(f"  Experimental missed points: {len(exp_missed)}")
    print(f"  Control missed points: {len(cont_missed)}")
    print(f"  Total score: {simulation_results[0]['experimental']['score'] + simulation_results[0]['control']['score']}")
    
    if exp_missed:
        print(f"\n🔍 Sample Experimental Missed Points:")
        for i, miss in enumerate(exp_missed[:5]):  # Show first 5
            print(f"    {miss['neuron']:8s} | {miss['t_start']:4d}-{miss['t_end']:4d}ms | "
                  f"Wanted:{miss['wanted']} Got:{miss['spikes']}")
        if len(exp_missed) > 5:
            print(f"    ... and {len(exp_missed) - 5} more")
    
    print(f"\n🎛️ Launching DNA Browser...")
    print("Features to test:")
    print("  • Sort dropdown: Change DNA ordering")
    print("  • Show dropdown: Switch between voltage traces, network graph, or both") 
    print("  • Condition dropdown: View experimental, control, or both conditions")
    print("  • DNA slider: Browse through different DNA solutions")
    print("  • 🎯 Gold markers: Criteria neurons highlighted")
    print("  • 🟠 Orange highlights: Missed scoring periods")
    print("  • 📝 Annotations: Expected vs actual spike counts")
    
    # Create the browser
    browser = create_dual_dna_browser(
        target_dnas=target_dnas,
        pruned_results=[],  # Empty since we're using target_dnas
        simulation_results=simulation_results,
        pruning_threshold=975,
        max_pruned_weights=20
    )
    
    return browser

def test_voltage_plot_directly():
    """Test the voltage plot function directly with new_jh_weights data."""
    print("\n🧪 Testing Voltage Plot Directly")
    print("=" * 30)
    
    # Create test data
    dna_vector = convert_weights_to_dna_vector(new_jh_weights, ACTIVE_SYNAPSES)
    dna_info = create_mock_dna_info(1, dna_vector)
    simulation_result = create_mock_simulation_results(dna_vector)
    
    print("📈 Creating voltage plot with missed scoring...")
    
    # Test experimental condition with missed scoring
    voltage_plot = create_voltage_plot(
        results=simulation_result,
        dna_info=dna_info,
        condition="experimental",
        show_missed_scoring=True
    )
    
    print("✅ Voltage plot created successfully!")
    print("   Look for:")
    print("   • 🎯 symbols next to criteria neurons")
    print("   • Orange highlighted regions for missed points") 
    print("   • 'Miss: W{wanted} G{got}' annotations")
    print("   • Total missed points in the title")
    
    return voltage_plot

if __name__ == "__main__":
    print("🚀 DNA Browser Test Suite")
    print("=" * 40)
    
    # Test 1: Direct voltage plot
    try:
        voltage_plot = test_voltage_plot_directly()
        print("✅ Voltage plot test completed")
    except Exception as e:
        print(f"❌ Voltage plot test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 40)
    
    # Test 2: Full DNA browser
    try:
        browser = test_dna_browser_with_new_jh_weights()
        print("✅ DNA browser test setup completed")
        print("\n🎉 Test successful! The DNA browser should now be displayed above.")
        print("   Try changing the dropdown settings to explore different visualizations.")
    except Exception as e:
        print(f"❌ DNA browser test failed: {e}")
        import traceback
        traceback.print_exc()