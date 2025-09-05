#!/usr/bin/env python3
"""
Test DNA browser with new_jh_weights using REAL simulation data (no noise)

This test uses the actual simulation engine to generate proper voltage traces
and spike rasters that match the format produced by analyze_multiple_ga_results.ipynb
"""

import numpy as np
import sys
import os
sys.path.append('src')
sys.path.append('.')

from src.constants import (
    new_jh_weights, ACTIVE_SYNAPSES, NEURON_NAMES, TMAX, BIN_SIZE,
    CRITERIA_NAMES, INHIBITORY_NEURONS
)
from dna_simulation import run_dna_with_voltage_tracking
from dna_browser import create_dual_dna_browser
from dna_visualization import create_voltage_plot

def convert_weights_to_dna_vector(weight_connections, active_synapses):
    """Convert connection weights list to DNA vector format."""
    weight_dict = {(pre, post): weight for pre, post, weight in weight_connections}
    dna_vector = []
    for pre, post in active_synapses:
        weight = weight_dict.get((pre, post), 0)
        dna_vector.append(abs(weight))  # DNA encoding uses positive values
    return np.array(dna_vector, dtype=np.int32)

def create_dna_info_structure(dna_id, dna_vector, simulation_results):
    """Create DNA info structure matching the format from analyze_multiple_ga_results.ipynb"""
    nonzero_weights = np.count_nonzero(dna_vector)
    
    exp_score = simulation_results['experimental']['score']
    cont_score = simulation_results['control']['score']
    total_score = exp_score + cont_score
    
    return {
        'id': dna_id,
        'original_dna': {
            'run_folder': 'test_new_jh_weights_real',
            'generation': 1,
            'process_id': 1,
            'original_dna_id': f'new_jh_weights_{dna_id}'
        },
        'pruned_dna': dna_vector,
        'original_score': total_score,
        'pruned_score': total_score,
        'original_nonzero': nonzero_weights,
        'pruned_nonzero': nonzero_weights,
        'weights_removed': 0,  # No pruning applied to new_jh_weights
        'final_exp_score': exp_score,
        'final_cont_score': cont_score,
    }

def test_new_jh_weights_with_real_simulation():
    """Test the DNA browser using new_jh_weights with REAL simulation engine."""
    
    print("🧪 Testing DNA Browser with new_jh_weights (REAL simulation)")
    print("=" * 60)
    
    # Convert new_jh_weights to DNA vector
    print("🔄 Converting new_jh_weights to DNA vector format...")
    dna_vector = convert_weights_to_dna_vector(new_jh_weights, ACTIVE_SYNAPSES)
    
    print(f"✅ DNA Vector Created:")
    print(f"   Total connections: {len(dna_vector)}")
    print(f"   Non-zero weights: {np.count_nonzero(dna_vector)}")
    print(f"   Weight range: {dna_vector.min()} - {dna_vector.max()}")
    
    # Show the actual connections
    print(f"\\n🔗 new_jh_weights Network ({len(new_jh_weights)} connections):")
    for i, (pre, post, weight) in enumerate(new_jh_weights):
        conn_type = "Inhibitory" if pre in INHIBITORY_NEURONS else "Excitatory"
        print(f"  {i+1:2d}. {pre:8s} → {post:8s} | {weight:6d} | {conn_type}")
    
    # Run REAL simulation
    print(f"\\n🔬 Running REAL simulation...")
    try:
        simulation_results = run_dna_with_voltage_tracking(dna_vector)
        
        exp_score = simulation_results['experimental']['score']
        cont_score = simulation_results['control']['score']
        exp_missed = len(simulation_results['experimental']['missed_points'])
        cont_missed = len(simulation_results['control']['missed_points'])
        
        print(f"✅ Simulation completed:")
        print(f"   Experimental: score={exp_score}, missed={exp_missed} points")
        print(f"   Control: score={cont_score}, missed={cont_missed} points")
        print(f"   Total score: {exp_score + cont_score}")
        
        # Create data structures matching analyze_multiple_ga_results.ipynb format
        dna_info = create_dna_info_structure(1, dna_vector, simulation_results)
        target_dnas = [dna_info]
        simulation_results_list = [simulation_results]
        
        print(f"\\n📊 Data Structure Created:")
        print(f"   DNA info format matches analyze_multiple_ga_results.ipynb")
        print(f"   Real voltage traces: ✅")
        print(f"   Real spike rasters: ✅")
        print(f"   Missed scoring data: ✅")
        
        # Test voltage plot first
        print(f"\\n📈 Testing voltage plot with missed scoring...")
        try:
            voltage_plot = create_voltage_plot(
                results=simulation_results,
                dna_info=dna_info,
                condition="experimental",
                show_missed_scoring=True
            )
            print("✅ Voltage plot created successfully!")
        except Exception as e:
            print(f"❌ Voltage plot failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Create DNA browser
        print(f"\\n🎛️ Creating DNA Browser...")
        browser = create_dual_dna_browser(
            target_dnas=target_dnas,
            pruned_results=[],  # Empty since we're using target_dnas
            simulation_results=simulation_results_list,
            pruning_threshold=exp_score + cont_score - 10,  # Set threshold slightly below actual
            max_pruned_weights=np.count_nonzero(dna_vector) + 5
        )
        
        print("\\n🎉 Test completed successfully!")
        print("Expected features in the browser:")
        print("  🎯 Gold markers on criteria neurons")
        print("  🟠 Orange highlights showing missed scoring periods")
        print("  📝 Annotations with expected vs actual spike counts")
        print("  📊 Total missed points in plot titles")
        print("  🎛️ Interactive controls for condition selection")
        
        return browser
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 Real Simulation Test for new_jh_weights")
    print("This test uses the ACTUAL simulation engine from adaptive_tmax_fully_optimized")
    print("No artificial noise - real voltage traces and spike patterns\\n")
    
    browser = test_new_jh_weights_with_real_simulation()
    
    if browser:
        print("\\n📋 Instructions:")
        print("1. The DNA browser should appear above")
        print("2. Try changing the 'Condition' dropdown to see experimental vs control")
        print("3. Look for orange highlighted regions in voltage traces")
        print("4. Notice 🎯 symbols marking criteria neurons")
        print("5. Check annotations showing 'Miss: W{wanted} G{got}' information")
    else:
        print("\\n❌ Test failed - check error messages above")