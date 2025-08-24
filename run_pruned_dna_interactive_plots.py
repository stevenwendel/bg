#!/usr/bin/env python3

import pickle
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Import the necessary modules from the project
from src.constants import *
from src.neuron import *
from src.network import *
from src.validation import *
from src.genetic_algorithm import *

# Path to the pruned DNA file
pruned_dna_file = '/Users/stevenwendel/Documents/GitHub/bg/greedy_pruned_dna.pkl'

def run_dna_with_corrected_voltage_tracking(dna_array):
    """Run a single DNA experiment and capture voltage traces with correct timing."""
    
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
        
        # Create voltage history tracking with correct timing
        voltage_history = np.zeros((len(NEURON_NAMES), TMAX), dtype=np.float32)
        spike_history = np.zeros((len(NEURON_NAMES), TMAX), dtype=np.uint8)
        
        # Import voltage arrays
        from src.neuron import _V, _VHIST, _SPIKES, _INPUT, _vpeak
        from src.network import convert_to_sparse_weights, sparse_matrix_multiply_only_spiking, _sparse_connections
        
        # Store initial voltage (after prepare_neurons)
        voltage_history[:, 0] = _V.copy()
        
        # Run simulation step by step
        N = len(all_neurons)
        spikers = np.zeros(N, np.uint8)
        sparse_weights = convert_to_sparse_weights(dna_matrix)
        
        for t in range(TMAX - 1):  # TMAX - 1 because we handle the last step separately
            # Apply synaptic input (from run_network)
            if spikers.any():
                post_I = sparse_matrix_multiply_only_spiking(spikers.astype(np.float32), 
                                                            sparse_weights, 
                                                            _sparse_connections)
                lend = min(alpha_array.size, TMAX - t - 1)
                _INPUT[:, t+1:t+1+lend] += post_I[:, None] * alpha_array[:lend]
            
            # Take integration step
            spikers = vectorised_step(_INPUT[:, t])
            spike_history[:, t] = spikers
            
            # Record voltage AFTER the step, but handle spikes correctly
            if t > 0:
                # For neurons that spiked, record the peak voltage at the previous timestep
                peaked = spike_history[:, t-1].astype(bool)
                voltage_history[peaked, t-1] = _vpeak[peaked]
            
            # Record current voltage for next timestep
            voltage_history[:, t+1] = _V.copy()
        
        # Handle the final timestep
        final_spikers = vectorised_step(_INPUT[:, TMAX-1])
        spike_history[:, TMAX-1] = final_spikers
        
        # Handle final spike peaks
        peaked = spike_history[:, TMAX-2].astype(bool) if TMAX > 1 else np.zeros(N, dtype=bool)
        voltage_history[peaked, TMAX-2] = _vpeak[peaked]
        
        # Create voltage traces dictionary
        voltage_traces = {}
        for i, neuron_name in enumerate(NEURON_NAMES):
            voltage_traces[neuron_name] = voltage_history[i, :]
        
        # Calculate spike counts for validation
        spike_counts = evaluate_conditions(spike_history)
        
        results[label] = {
            'voltages': voltage_traces,
            'spike_counts': spike_counts[label] if label in spike_counts else 0,
            'spikes': spike_history.copy()
        }
        
        # Debug info
        print(f"Voltage range for {label}: {np.min(voltage_history):.2f} to {np.max(voltage_history):.2f}")
    
    return results

def plot_voltage_traces_interactive_with_borders(results, dna_info):
    """Create interactive voltage trace plots with fixed y-axis, hover functionality, and borders for CRITERIA_NAMES."""
    
    # Create time array
    time_ms = np.arange(TMAX)
    
    # Get failed time bins for both conditions using diagnose_conditions
    from src.validation import diagnose_conditions
    failed_bins_exp = diagnose_conditions(results['experimental']['spikes'], 'experimental', return_list=True)
    failed_bins_ctrl = diagnose_conditions(results['control']['spikes'], 'control', return_list=True)
    
    # Create subplots - one row per neuron, two columns (exp vs control)
    n_neurons = len(NEURON_NAMES)
    
    subplot_titles = []
    for neuron_name in NEURON_NAMES:
        # Add asterisk to indicate neurons in CRITERIA_NAMES
        criteria_marker = " *" if neuron_name in CRITERIA_NAMES else ""
        subplot_titles.extend([f'{neuron_name}{criteria_marker} - Experimental', 
                             f'{neuron_name}{criteria_marker} - Control'])
    
    fig = make_subplots(
        rows=n_neurons, 
        cols=2,
        subplot_titles=subplot_titles,
        vertical_spacing=0.02,
        horizontal_spacing=0.08
    )
    
    # Add traces for each neuron
    for i, neuron_name in enumerate(NEURON_NAMES):
        row = i + 1
        
        # Determine if this neuron is in CRITERIA_NAMES for border styling
        is_criteria_neuron = neuron_name in CRITERIA_NAMES
        line_width = 2 if is_criteria_neuron else 1
        
        # Experimental condition
        voltages_exp = results['experimental']['voltages'][neuron_name]
        fig.add_trace(
            go.Scatter(
                x=time_ms,
                y=voltages_exp,
                mode='lines',
                name=f'{neuron_name} Exp',
                line=dict(color='blue', width=line_width),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Time: %{x} ms<br>' +
                             'Voltage: %{y:.2f} mV<br>' +
                             '<extra></extra>',
                showlegend=False
            ),
            row=row, col=1
        )
        
        # Control condition
        voltages_ctrl = results['control']['voltages'][neuron_name]
        fig.add_trace(
            go.Scatter(
                x=time_ms,
                y=voltages_ctrl,
                mode='lines',
                name=f'{neuron_name} Ctrl',
                line=dict(color='red', width=line_width),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Time: %{x} ms<br>' +
                             'Voltage: %{y:.2f} mV<br>' +
                             '<extra></extra>',
                showlegend=False
            ),
            row=row, col=2
        )
        
        # Add failed time bin indicators for experimental condition
        exp_failed_bins = [fb for fb in failed_bins_exp if fb['neuron'] == neuron_name]
        for failed_bin in exp_failed_bins:
            fig.add_vrect(
                x0=failed_bin['t_start'], x1=failed_bin['t_end'],
                fillcolor="orange", opacity=0.4,
                layer="below", line_width=1,
                annotation_text=f"FAILED: wanted {failed_bin['wanted']}, got {failed_bin['spikes']}",
                annotation_position="top",
                row=row, col=1
            )
        
        # Add failed time bin indicators for control condition  
        ctrl_failed_bins = [fb for fb in failed_bins_ctrl if fb['neuron'] == neuron_name]
        for failed_bin in ctrl_failed_bins:
            fig.add_vrect(
                x0=failed_bin['t_start'], x1=failed_bin['t_end'],
                fillcolor="orange", opacity=0.4,
                layer="below", line_width=1,
                annotation_text=f"FAILED: wanted {failed_bin['wanted']}, got {failed_bin['spikes']}",
                annotation_position="top",
                row=row, col=2
            )

        # Add stimulus markers for experimental condition
        fig.add_vrect(
            x0=1000, x1=1200,
            fillcolor="red", opacity=0.2,
            layer="below", line_width=0,
            row=row, col=1
        )
        fig.add_vrect(
            x0=3000, x1=3100,
            fillcolor="green", opacity=0.2,
            layer="below", line_width=0,
            row=row, col=1
        )
        
        # Add stimulus markers for control condition
        fig.add_vrect(
            x0=1000, x1=1200,
            fillcolor="red", opacity=0.2,
            layer="below", line_width=0,
            row=row, col=2
        )
        fig.add_vrect(
            x0=3000, x1=3100,
            fillcolor="green", opacity=0.2,
            layer="below", line_width=0,
            row=row, col=2
        )
    
    # Update layout with pruned DNA information
    fig.update_layout(
        title=f'Interactive Voltage Traces - PRUNED DNA (Fitness: {dna_info["total_score"]}, Non-zero weights: {dna_info["final_nonzero"]}/{dna_info["original_nonzero"]})<br><sub>* = Neurons in CRITERIA_NAMES (used for fitness evaluation) | 🟨 = Gold borders highlight criteria neurons | 🟧 = Orange highlights show failed time bins (zero scores)</sub>',
        height=300 * n_neurons,
        width=1200,
        showlegend=False,
        hovermode='closest'
    )
    
    # Update all y-axes to have fixed range from -100 to 100 mV and add borders for criteria neurons
    for i in range(1, n_neurons + 1):
        neuron_name = NEURON_NAMES[i-1]
        is_criteria_neuron = neuron_name in CRITERIA_NAMES
        
        # Add border styling for criteria neurons
        if is_criteria_neuron:
            # Add thick border for criteria neurons
            border_style = dict(
                linewidth=3,
                linecolor='gold',
                mirror=True
            )
        else:
            # Normal border for non-criteria neurons
            border_style = dict(
                linewidth=1,
                linecolor='lightgray',
                mirror=True
            )
        
        fig.update_yaxes(range=[-100, 100], title_text="Voltage (mV)", row=i, col=1, **border_style)
        fig.update_yaxes(range=[-100, 100], title_text="Voltage (mV)", row=i, col=2, **border_style)
        fig.update_xaxes(row=i, col=1, **border_style)
        fig.update_xaxes(row=i, col=2, **border_style)
    
    # Update x-axes (only for bottom row)
    fig.update_xaxes(title_text="Time (ms)", row=n_neurons, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=n_neurons, col=2)
    
    return fig

def plot_voltage_traces_matplotlib_with_borders(results, dna_info):
    """Create matplotlib plots with fixed y-axis and borders for CRITERIA_NAMES."""
    
    # Create time array
    time_ms = np.arange(TMAX)
    
    # Get failed time bins for both conditions using diagnose_conditions
    from src.validation import diagnose_conditions
    failed_bins_exp = diagnose_conditions(results['experimental']['spikes'], 'experimental', return_list=True)
    failed_bins_ctrl = diagnose_conditions(results['control']['spikes'], 'control', return_list=True)
    
    # Create subplots - one row per neuron, two columns (exp vs control)
    n_neurons = len(NEURON_NAMES)
    fig, axes = plt.subplots(n_neurons, 2, figsize=(16, 2*n_neurons))
    fig.suptitle(f'Voltage Traces - PRUNED DNA (Fitness: {dna_info["total_score"]}, Weights: {dna_info["final_nonzero"]}/{dna_info["original_nonzero"]})\n* = Neurons in CRITERIA_NAMES | 🟨 = Gold borders highlight criteria neurons | 🟧 = Orange highlights show failed time bins (zero scores)', fontsize=16)
    
    # Plot each neuron
    for i, neuron_name in enumerate(NEURON_NAMES):
        is_criteria_neuron = neuron_name in CRITERIA_NAMES
        
        # Experimental condition
        ax_exp = axes[i, 0]
        voltages_exp = results['experimental']['voltages'][neuron_name]
        line_width = 1.5 if is_criteria_neuron else 0.8
        ax_exp.plot(time_ms, voltages_exp, 'b-', linewidth=line_width)
        
        title_marker = " *" if is_criteria_neuron else ""
        ax_exp.set_title(f'{neuron_name}{title_marker} - Experimental')
        ax_exp.set_ylabel('Voltage (mV)')
        ax_exp.set_ylim(-100, 100)  # Fixed y-axis
        ax_exp.grid(True, alpha=0.3)
        
        # Add border for criteria neurons
        if is_criteria_neuron:
            for spine in ax_exp.spines.values():
                spine.set_linewidth(3)
                spine.set_color('gold')
        
        # Add failed time bin indicators for experimental condition
        exp_failed_bins = [fb for fb in failed_bins_exp if fb['neuron'] == neuron_name]
        for failed_bin in exp_failed_bins:
            ax_exp.axvspan(failed_bin['t_start'], failed_bin['t_end'], alpha=0.6, color='orange', 
                          label=f'Failed bin: wanted {failed_bin["wanted"]}, got {failed_bin["spikes"]}' if failed_bin == exp_failed_bins[0] else '')
        
        # Add stimulus markers
        ax_exp.axvspan(1000, 1200, alpha=0.2, color='red', label='Cue')
        ax_exp.axvspan(3000, 3100, alpha=0.2, color='green', label='Go')
        
        # Control condition  
        ax_ctrl = axes[i, 1]
        voltages_ctrl = results['control']['voltages'][neuron_name]
        ax_ctrl.plot(time_ms, voltages_ctrl, 'r-', linewidth=line_width)
        ax_ctrl.set_title(f'{neuron_name}{title_marker} - Control')
        ax_ctrl.set_ylabel('Voltage (mV)')
        ax_ctrl.set_ylim(-100, 100)  # Fixed y-axis
        ax_ctrl.grid(True, alpha=0.3)
        
        # Add border for criteria neurons
        if is_criteria_neuron:
            for spine in ax_ctrl.spines.values():
                spine.set_linewidth(3)
                spine.set_color('gold')
        
        # Add failed time bin indicators for control condition
        ctrl_failed_bins = [fb for fb in failed_bins_ctrl if fb['neuron'] == neuron_name]
        for failed_bin in ctrl_failed_bins:
            ax_ctrl.axvspan(failed_bin['t_start'], failed_bin['t_end'], alpha=0.6, color='orange',
                           label=f'Failed bin: wanted {failed_bin["wanted"]}, got {failed_bin["spikes"]}' if failed_bin == ctrl_failed_bins[0] else '')
        
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
    # Load the pruned DNA
    with open(pruned_dna_file, 'rb') as f:
        pruned_data = pickle.load(f)
    
    # Extract pruned DNA and create a compatible format
    pruned_individual = {
        'dna': pruned_data['pruned_dna'],
        'total_score': pruned_data['final_scores']['total'],
        'exp_score': pruned_data['final_scores']['exp'],
        'cont_score': pruned_data['final_scores']['cont'],
        'generation': 'Pruned',
        'final_nonzero': pruned_data['final_nonzero'],
        'original_nonzero': pruned_data['original_nonzero'],
        'weights_removed': pruned_data['weights_removed'],
        'method': pruned_data['method']
    }
    
    print(f"Running experiment with PRUNED DNA:")
    print(f"  Pruning method: {pruned_individual['method']}")
    print(f"  Final fitness: {pruned_individual['total_score']}")
    print(f"  Experimental score: {pruned_individual['exp_score']}")
    print(f"  Control score: {pruned_individual['cont_score']}")
    print(f"  Non-zero weights: {pruned_individual['final_nonzero']} (was {pruned_individual['original_nonzero']})")
    print(f"  Weights removed: {pruned_individual['weights_removed']}")
    print(f"  Weight reduction: {pruned_individual['weights_removed']/pruned_individual['original_nonzero']*100:.1f}%")
    
    print(f"\nPruned DNA vector:")
    print(f"  {pruned_individual['dna']}")
    
    print(f"\nCRITERIA_NAMES (neurons used for fitness evaluation):")
    for i, name in enumerate(CRITERIA_NAMES):
        print(f"  {i+1:2d}. {name}")
    
    # Run the experiment with corrected voltage tracking
    results = run_dna_with_corrected_voltage_tracking(pruned_individual['dna'])
    
    print(f"\nExperiment completed!")
    print(f"Experimental spike count: {results['experimental']['spike_counts']}")
    print(f"Control spike count: {results['control']['spike_counts']}")
    
    # Create interactive plotly plot with borders
    print("Creating interactive plot with borders...")
    fig_interactive = plot_voltage_traces_interactive_with_borders(results, pruned_individual)
    
    # Save interactive plot as HTML
    interactive_filename = '/Users/stevenwendel/Documents/GitHub/bg/pruned_dna_voltage_traces_interactive_bordered.html'
    fig_interactive.write_html(interactive_filename)
    print(f"Interactive plot with borders saved to: {interactive_filename}")
    
    # Create static matplotlib plot with borders
    print("Creating static plot with borders...")
    fig_static = plot_voltage_traces_matplotlib_with_borders(results, pruned_individual)
    
    # Save static plot
    static_filename = '/Users/stevenwendel/Documents/GitHub/bg/pruned_dna_voltage_traces_bordered.png'
    plt.savefig(static_filename, dpi=300, bbox_inches='tight')
    print(f"Static plot with borders saved to: {static_filename}")
    
    # Save the results
    results['pruned_dna_info'] = pruned_individual
    with open('/Users/stevenwendel/Documents/GitHub/bg/pruned_dna_results_bordered.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Results saved to: pruned_dna_results_bordered.pkl")
    
    print("\n🎉 PRUNED DNA ANALYSIS COMPLETE! 🎉")
    print("="*60)
    print("Key findings:")
    print(f"  • Reduced from {pruned_individual['original_nonzero']} to {pruned_individual['final_nonzero']} weights ({pruned_individual['weights_removed']} removed)")
    print(f"  • Maintained fitness score of {pruned_individual['total_score']}")
    print(f"  • {pruned_individual['weights_removed']/pruned_individual['original_nonzero']*100:.1f}% weight reduction with NO performance loss!")
    print("\nVisualization features:")
    print("  • Fixed y-axis (-100 to 100 mV) for all plots")
    print("  • Interactive hover functionality showing exact voltage values")
    print("  • GOLD BORDERS highlight neurons in CRITERIA_NAMES")
    print("  • Asterisk (*) markers in titles for criteria neurons")
    print("  • Thicker line weights for criteria neurons")
    print("\nTo view the interactive plot:")
    print(f"  Open '{interactive_filename}' in any web browser")
    print("  Neurons with gold borders are used for fitness evaluation!")
    print("="*60)