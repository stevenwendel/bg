#!/usr/bin/env python3
"""
Simple test to verify criteria interval highlighting works
"""

import sys
import numpy as np
sys.path.append('src')
sys.path.append('.')

from src.constants import NEURON_NAMES, CRITERIA_NAMES, CRITERIA, EPOCHS, TMAX
from src.viz import plot_neurons_interactive
from src.network import create_experiment

def test_criteria_highlighting():
    """Test criteria interval highlighting functionality."""
    print("🧪 Testing Criteria Interval Highlighting")
    print("=" * 50)
    
    # Print the CRITERIA structure for reference
    print("📋 CRITERIA Structure:")
    for condition in ['experimental', 'control']:
        print(f"\n{condition.upper()} Condition:")
        for neuron in CRITERIA_NAMES:
            criteria_info = CRITERIA.get(condition, {}).get(neuron, {})
            if criteria_info:
                interval = criteria_info['interval']
                io_state = criteria_info['io']
                print(f"  {neuron:8s}: {interval[0]:4d}-{interval[1]:4d}ms ({io_state})")
    
    print(f"\n📊 EPOCHS Reference:")
    for epoch_name, interval in EPOCHS.items():
        print(f"  {epoch_name:8s}: {interval[0]:4d}-{interval[1]:4d}ms")
    
    # Create simple test voltage data
    print("\n🔬 Creating test voltage data...")
    hist_Vs = []
    hist_us = []
    for i in range(len(NEURON_NAMES)):
        # Simple voltage trace with some spikes
        voltage = np.full(TMAX, -60.0, dtype=np.float32)
        # Add some random spikes
        spike_times = np.random.choice(TMAX, size=20, replace=False)
        for spike_time in spike_times:
            if spike_time + 10 < TMAX:
                voltage[spike_time:spike_time+10] = np.linspace(-60, 35, 10)
        
        hist_Vs.append(voltage)
        hist_us.append(np.zeros(TMAX))
    
    hist_Vs = np.array(hist_Vs)
    hist_us = np.array(hist_us)
    
    # Get stimulus waves
    splits, input_waves, alpha_array = create_experiment()
    cue_wave, go_wave = input_waves
    
    print("\n📈 Testing EXPERIMENTAL condition plot...")
    try:
        plot_neurons_interactive(
            hist_Vs=hist_Vs,
            hist_us=hist_us,
            neuron_names=NEURON_NAMES,
            sq_wave=cue_wave,
            go_wave=go_wave,
            show_u=False,
            title="Test: Criteria Interval Highlighting - EXPERIMENTAL",
            condition="experimental"
        )
        print("✅ Experimental plot created!")
    except Exception as e:
        print(f"❌ Experimental plot failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n📈 Testing CONTROL condition plot...")
    try:
        plot_neurons_interactive(
            hist_Vs=hist_Vs,
            hist_us=hist_us,
            neuron_names=NEURON_NAMES,
            sq_wave=cue_wave,
            go_wave=go_wave,
            show_u=False,
            title="Test: Criteria Interval Highlighting - CONTROL",
            condition="control"
        )
        print("✅ Control plot created!")
    except Exception as e:
        print(f"❌ Control plot failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Expected Features:")
    print("  🟠 Light orange background regions for criteria intervals")
    print("  📝 'Criteria: ON/OFF' annotations")
    print("  🎯 Gold markers for criteria neurons")
    print("  📊 Different intervals for experimental vs control conditions")
    
    print("\n📋 Key Intervals to Look For:")
    print("  Somat (experimental): 1000-2000ms (ON) - sample period")
    print("  ALMprep (experimental): 1000-3200ms (ON) - sample + delay")
    print("  ALMresp (experimental): 3000-5000ms (ON) - response period")
    print("  SNR neurons: Different patterns in experimental vs control")

if __name__ == "__main__":
    test_criteria_highlighting()