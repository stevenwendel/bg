#!/usr/bin/env python3
"""
Final test demonstrating all enhanced visualization features:
1. Criteria interval highlighting (light orange background)
2. Missed scoring visualization (darker orange highlights)
3. Real simulation data from new_jh_weights
"""

import sys
import numpy as np
sys.path.append('src')
sys.path.append('.')

from src.constants import new_jh_weights, ACTIVE_SYNAPSES, NEURON_NAMES, CRITERIA_NAMES, CRITERIA, EPOCHS
from dna_simulation import run_dna_with_voltage_tracking
from dna_visualization import create_voltage_plot

def convert_weights_to_dna_vector(weight_connections, active_synapses):
    """Convert connection weights list to DNA vector format."""
    weight_dict = {(pre, post): weight for pre, post, weight in weight_connections}
    dna_vector = []
    for pre, post in active_synapses:
        weight = weight_dict.get((pre, post), 0)
        dna_vector.append(abs(weight))
    return np.array(dna_vector, dtype=np.int32)

def create_dna_info(dna_id, dna_vector, simulation_results):
    """Create DNA info structure."""
    exp_score = simulation_results['experimental']['score']
    cont_score = simulation_results['control']['score']
    
    return {
        'id': dna_id,
        'original_dna': {'run_folder': 'new_jh_weights_final_test'},
        'pruned_dna': dna_vector,
        'pruned_score': exp_score + cont_score,
        'original_nonzero': np.count_nonzero(dna_vector),
        'pruned_nonzero': np.count_nonzero(dna_vector),
        'weights_removed': 0,
        'final_exp_score': exp_score,
        'final_cont_score': cont_score,
    }

def main():
    print("🎉 Final Enhanced Visualization Test")
    print("=" * 50)
    print("Features being tested:")
    print("  🟠 Criteria interval highlighting (light orange background)")
    print("  🟠 Missed scoring visualization (darker orange highlights)")
    print("  📝 Combined annotations (criteria states + missed points)")
    print("  ⚡ Real simulation data (no artificial noise)")
    print("  🎯 Gold markers for criteria neurons")
    
    # Convert new_jh_weights and run real simulation
    print("\n🔄 Setting up new_jh_weights simulation...")
    dna_vector = convert_weights_to_dna_vector(new_jh_weights, ACTIVE_SYNAPSES)
    print(f"  DNA vector: {np.count_nonzero(dna_vector)} non-zero weights")
    
    try:
        print("🔬 Running REAL simulation...")
        simulation_results = run_dna_with_voltage_tracking(dna_vector)
        dna_info = create_dna_info(1, dna_vector, simulation_results)
        
        exp_missed = len(simulation_results['experimental']['missed_points'])
        cont_missed = len(simulation_results['control']['missed_points'])
        
        print(f"✅ Simulation complete:")
        print(f"  Exp: {simulation_results['experimental']['score']} (missed: {exp_missed})")
        print(f"  Cont: {simulation_results['control']['score']} (missed: {cont_missed})")
        
        # Show criteria structure for reference
        print(f"\n📋 Key Criteria Intervals (Experimental):")
        for neuron in ['Somat', 'ALMprep', 'ALMresp', 'VMprep', 'VMresp']:
            if neuron in CRITERIA['experimental']:
                criteria = CRITERIA['experimental'][neuron]
                interval = criteria['interval']
                state = criteria['io']
                print(f"  {neuron:8s}: {interval[0]:4d}-{interval[1]:4d}ms ({state.upper()})")
        
        print("\n📈 Creating EXPERIMENTAL condition plot...")
        create_voltage_plot(
            results=simulation_results,
            dna_info=dna_info,
            condition="experimental",
            show_missed_scoring=True
        )
        print("✅ Experimental plot created!")
        
        print("\n📈 Creating CONTROL condition plot...")
        create_voltage_plot(
            results=simulation_results,
            dna_info=dna_info,
            condition="control",
            show_missed_scoring=True
        )
        print("✅ Control plot created!")
        
        print("\n🎉 SUCCESS! Enhanced visualization features:")
        print("  ✅ Criteria intervals highlighted (light orange background)")
        print("  ✅ Missed scoring regions highlighted (darker orange)")
        print("  ✅ Combined annotations showing both criteria and misses")
        print("  ✅ Different patterns for experimental vs control")
        print("  ✅ Real simulation data (authentic neural dynamics)")
        
        print("\n📊 Compare the two plots to see:")
        print("  • Different criteria intervals between conditions")
        print("  • Where actual neural activity matches/misses expectations")
        print("  • How new_jh_weights network performs in both conditions")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()