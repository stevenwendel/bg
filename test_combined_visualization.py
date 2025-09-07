#!/usr/bin/env python3
"""
Test combined visualization: criteria highlighting + missed scoring
"""

import sys
import numpy as np
sys.path.append('src')
sys.path.append('.')

from src.constants import NEURON_NAMES, CRITERIA_NAMES, TMAX
from src.viz import plot_neurons_interactive
from src.network import create_experiment
from src.validation import diagnose_conditions

def test_combined_features():
    """Test both criteria highlighting and missed scoring together."""
    print("🧪 Testing Combined Visualization Features")
    print("=" * 50)
    
    # Create test voltage data with some realistic spikes
    hist_Vs = []
    hist_us = []
    spike_raster = np.zeros((len(NEURON_NAMES), TMAX), dtype=np.uint8)
    
    for i, neuron_name in enumerate(NEURON_NAMES):
        voltage = np.full(TMAX, -60.0, dtype=np.float32)
        
        # Add some spikes that might miss criteria
        if neuron_name == "Somat":
            # Add spikes outside the criteria window (should miss)
            spike_times = [500, 2500, 4500]  # Outside 1000-2000ms experimental criteria
        elif neuron_name == "ALMprep":
            # Add some spikes within criteria window
            spike_times = [1200, 1800, 2500]  # Some within 1000-3200ms criteria
        elif neuron_name == "ALMresp":
            # Add spikes before response period (should miss)
            spike_times = [2000, 2500]  # Before 3000ms response criteria
        else:
            # Random sparse spikes for other neurons
            spike_times = np.random.choice(TMAX, size=3, replace=False)
        
        # Create voltage spikes
        for spike_time in spike_times:
            if spike_time + 15 < TMAX:
                voltage[spike_time:spike_time+15] = np.linspace(-60, 35, 15)
                spike_raster[i, spike_time] = 1
        
        hist_Vs.append(voltage)
        hist_us.append(np.zeros(TMAX))
    
    hist_Vs = np.array(hist_Vs)
    hist_us = np.array(hist_us)
    
    # Get stimulus waves
    splits, input_waves, alpha_array = create_experiment()
    cue_wave, go_wave = input_waves
    
    print("🔍 Analyzing missed scoring...")
    try:
        exp_missed = diagnose_conditions(spike_raster, "experimental", return_list=True)
        print(f"  Experimental missed points: {len(exp_missed)}")
        if exp_missed:
            for miss in exp_missed[:3]:
                print(f"    {miss['neuron']:8s} | {miss['t_start']:4d}-{miss['t_end']:4d}ms | W:{miss['wanted']} G:{miss['spikes']}")
    except Exception as e:
        print(f"  ❌ Missed scoring analysis failed: {e}")
        exp_missed = []
    
    print("\n📈 Creating combined visualization...")
    try:
        plot_neurons_interactive(
            hist_Vs=hist_Vs,
            hist_us=hist_us,
            neuron_names=NEURON_NAMES,
            sq_wave=cue_wave,
            go_wave=go_wave,
            show_u=False,
            title=f"Combined Test: Criteria + Missed Scoring ({len(exp_missed)} missed points)",
            spike_raster=spike_raster,
            condition="experimental",
            show_missed_scoring=True
        )
        print("✅ Combined visualization created!")
        
        print("\n🎉 Expected Combined Features:")
        print("  🟠 Light orange background: Criteria intervals (15% opacity)")
        print("  🟠 Darker orange highlights: Missed scoring periods (30% opacity)")
        print("  📝 'Criteria: ON/OFF' annotations")
        print("  📝 'Miss: W{wanted} G{got}' annotations")
        print("  🎯 Gold markers for criteria neurons")
        print("  📊 Total missed points in title")
        
    except Exception as e:
        print(f"❌ Combined visualization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_combined_features()