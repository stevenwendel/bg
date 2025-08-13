#!/usr/bin/env python3

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Import the necessary modules from the project
from src.constants import *
from src.neuron import *
from src.network import *
from src.validation import *
from src.genetic_algorithm import *

def run_dna_with_voltage_tracking(dna_array):
    """Run a single DNA experiment and capture voltage traces for all neurons."""
    
    # Convert DNA to weight matrix
    dna_matrix = decode_dna_to_matrix(dna_array)
    
    # Create experimental setup
    all_neurons = create_neurons()
    splits, input_waves, alpha_array = create_experiment()
    cue_wave, go_wave = input_waves
    
    results = {}
    
    # Run both experimental and control conditions
    for label, ctl in [("experimental", False), ("control", True)]:
        print(f"Running {label} condition...")
        
        # Prepare neurons for this condition
        prepare_neurons(all_neurons, cue_wave, go_wave, ctl)
        
        # Reset time pointer
        import src.neuron as neuron_module
        neuron_module.t_pointer = 0
        
        # Run the network simulation
        run_network(all_neurons, dna_matrix, alpha_array)
        
        # Capture voltage histories  
        from src.neuron import _VHIST, _SPIKES
        voltage_traces = {}
        for i, neuron_name in enumerate(NEURON_NAMES):
            voltage_traces[neuron_name] = _VHIST[i, :].copy()
        
        # Calculate spike counts for validation
        spike_counts = evaluate_conditions(_SPIKES)
        
        results[label] = {
            'voltages': voltage_traces,
            'spike_counts': spike_counts[label] if label in spike_counts else 0,
            'spikes': _SPIKES.copy()
        }
    
    return results

def plot_voltage_traces(results, dna_info):
    """Plot voltage traces for all neurons in both conditions."""
    
    # Create time array
    time_ms = np.arange(TMAX)
    
    # Create subplots - one row per neuron, two columns (exp vs control)
    n_neurons = len(NEURON_NAMES)
    fig, axes = plt.subplots(n_neurons, 2, figsize=(16, 2*n_neurons))
    fig.suptitle(f'Voltage Traces - Best DNA (Fitness: {dna_info["total_score"]})', fontsize=16)
    
    # Plot each neuron
    for i, neuron_name in enumerate(NEURON_NAMES):
        # Experimental condition
        ax_exp = axes[i, 0]
        voltages_exp = results['experimental']['voltages'][neuron_name]
        ax_exp.plot(time_ms, voltages_exp, 'b-', linewidth=0.8)
        ax_exp.set_title(f'{neuron_name} - Experimental')
        ax_exp.set_ylabel('Voltage (mV)')
        ax_exp.grid(True, alpha=0.3)
        
        # Add stimulus markers
        ax_exp.axvspan(1000, 1200, alpha=0.2, color='red', label='Cue')
        ax_exp.axvspan(3000, 3100, alpha=0.2, color='green', label='Go')
        
        # Control condition  
        ax_ctrl = axes[i, 1]
        voltages_ctrl = results['control']['voltages'][neuron_name]
        ax_ctrl.plot(time_ms, voltages_ctrl, 'r-', linewidth=0.8)
        ax_ctrl.set_title(f'{neuron_name} - Control')
        ax_ctrl.set_ylabel('Voltage (mV)')
        ax_ctrl.grid(True, alpha=0.3)
        
        # Add stimulus markers
        ax_ctrl.axvspan(1000, 1200, alpha=0.2, color='red', label='Cue')
        ax_ctrl.axvspan(3000, 3100, alpha=0.2, color='green', label='Go')
        
        # Only add x-label to bottom row
        if i == n_neurons - 1:
            ax_exp.set_xlabel('Time (ms)')
            ax_ctrl.set_xlabel('Time (ms)')
    
    # Add legend to first subplot
    axes[0, 0].legend(loc='upper right', fontsize='small')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Load the best DNA
    with open('/Users/stevenwendel/Documents/GitHub/bg/best_dna.pkl', 'rb') as f:
        best_individual = pickle.load(f)
    
    print(f"Running experiment with best DNA:")
    print(f"  Fitness: {best_individual['total_score']}")
    print(f"  Experimental score: {best_individual['exp_score']}")
    print(f"  Control score: {best_individual['cont_score']}")
    print(f"  From generation: {best_individual['generation']}")
    
    # Run the experiment with voltage tracking
    results = run_dna_with_voltage_tracking(best_individual['dna'])
    
    print(f"Experiment completed!")
    print(f"Experimental spike count: {results['experimental']['spike_counts']}")
    print(f"Control spike count: {results['control']['spike_counts']}")
    
    # Create and save the plot
    fig = plot_voltage_traces(results, best_individual)
    
    # Save the plot
    plt.savefig('/Users/stevenwendel/Documents/GitHub/bg/best_dna_voltage_traces.png', 
                dpi=300, bbox_inches='tight')
    print("Voltage traces plot saved to: best_dna_voltage_traces.png")
    
    # Save the results
    with open('/Users/stevenwendel/Documents/GitHub/bg/best_dna_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Results saved to: best_dna_results.pkl")
    
    # Show the plot
    plt.show()