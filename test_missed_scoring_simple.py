#!/usr/bin/env python3
"""
Simple test to verify missed scoring visualization works with new_jh_weights
"""

import sys
import numpy as np
sys.path.append('src')
sys.path.append('.')

from src.constants import new_jh_weights, ACTIVE_SYNAPSES, NEURON_NAMES, CRITERIA_NAMES, TMAX
from src.validation import diagnose_conditions
from src.viz import plot_neurons_interactive
from src.network import create_experiment

def simple_test():
    """Simple test of missed scoring functionality."""
    print("🧪 Simple Missed Scoring Test")
    print("=" * 40)
    
    # Create simple test data
    print("📊 Creating test spike raster...")
    spike_raster = np.zeros((len(NEURON_NAMES), TMAX), dtype=np.uint8)
    
    # Add some test spikes that will miss the criteria
    for i, neuron_name in enumerate(NEURON_NAMES):
        if neuron_name in CRITERIA_NAMES:
            # Add sparse spikes that will likely miss criteria
            spike_times = np.random.choice(TMAX, size=10, replace=False)
            spike_raster[i, spike_times] = 1
    
    print("🔍 Testing diagnose_conditions...")
    # Test the diagnose_conditions function directly
    try:
        exp_missed = diagnose_conditions(spike_raster, "experimental", return_list=True)
        cont_missed = diagnose_conditions(spike_raster, "control", return_list=True)
        
        print(f"✅ Experimental missed points: {len(exp_missed)}")
        print(f"✅ Control missed points: {len(cont_missed)}")
        
        if exp_missed:
            print("Sample missed points:")
            for miss in exp_missed[:3]:
                print(f"  {miss['neuron']:8s} | {miss['t_start']:4d}-{miss['t_end']:4d}ms | W:{miss['wanted']} G:{miss['spikes']}")
        
    except Exception as e:
        print(f"❌ diagnose_conditions failed: {e}")
        return False
    
    print("\\n📈 Testing voltage plot with missed scoring...")
    
    # Create test voltage data
    hist_Vs = []
    hist_us = []
    for i in range(len(NEURON_NAMES)):
        voltage = np.random.normal(-60, 5, TMAX)  # Random voltage around -60mV
        hist_Vs.append(voltage)
        hist_us.append(np.zeros(TMAX))  # Dummy u values
    
    hist_Vs = np.array(hist_Vs)
    hist_us = np.array(hist_us)
    
    # Get stimulus waves
    splits, input_waves, alpha_array = create_experiment()
    cue_wave, go_wave = input_waves
    
    try:
        # Test the enhanced plotting function
        plot_neurons_interactive(
            hist_Vs=hist_Vs,
            hist_us=hist_us, 
            neuron_names=NEURON_NAMES,
            sq_wave=cue_wave,
            go_wave=go_wave,
            show_u=False,
            title="Test: new_jh_weights Missed Scoring Visualization",
            spike_raster=spike_raster,
            condition="experimental",
            show_missed_scoring=True
        )
        
        print("✅ Plot created successfully!")
        print("Expected features:")
        print("  🎯 Gold markers on criteria neurons")
        print("  🟠 Orange highlights on missed scoring periods")
        print("  📝 'Miss: W{wanted} G{got}' annotations")
        print("  📊 Total missed points in title")
        
        return True
        
    except Exception as e:
        print(f"❌ Plotting failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_test()
    if success:
        print("\\n🎉 Test completed successfully!")
    else:
        print("\\n❌ Test failed!")