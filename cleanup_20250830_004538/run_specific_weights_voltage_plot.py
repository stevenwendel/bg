#!/usr/bin/env python3
"""
Script to run specific connection weights and generate voltage trace plots.

This script takes the new_jh_weights from constants.py and creates:
1. A DNA vector from the specified connections
2. Runs both experimental and control simulations
3. Generates interactive voltage trace plots similar to analyze_multiple_ga_results.ipynb

Usage:
    python run_specific_weights_voltage_plot.py
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path

# Import project modules
from src.constants import *
from src.neuron import *
from src.network import *
from src.validation import evaluate_conditions, diagnose_conditions
from src.genetic_algorithm import decode_dna_to_matrix

def create_dna_from_weights(weight_connections):
    """
    Create a DNA vector from specific weight connections.
    
    Args:
        weight_connections: List of tuples (from_neuron, to_neuron, weight)
    
    Returns:
        numpy array representing the DNA vector
    """
    # Initialize DNA with zeros
    dna = np.zeros(len(ACTIVE_SYNAPSES), dtype=np.int32)
    
    # Map connections to DNA indices
    connection_map = {}
    for i, (from_neuron, to_neuron) in enumerate(ACTIVE_SYNAPSES):
        connection_map[(from_neuron, to_neuron)] = i
    
    # Set weights in DNA
    connections_found = []
    connections_not_found = []
    
    for from_neuron, to_neuron, weight in weight_connections:
        if (from_neuron, to_neuron) in connection_map:
            idx = connection_map[(from_neuron, to_neuron)]
            dna[idx] = weight
            connections_found.append((from_neuron, to_neuron, weight))
        else:
            connections_not_found.append((from_neuron, to_neuron, weight))
    
    # Report results
    print(f"✅ Successfully mapped {len(connections_found)} connections to DNA")
    if connections_not_found:
        print(f"⚠️  Could not find {len(connections_not_found)} connections in ACTIVE_SYNAPSES:")
        for from_n, to_n, w in connections_not_found:
            print(f"    {from_n} → {to_n} (weight {w})")
    
    return dna, connections_found

def run_dna_with_voltage_tracking(dna_array):
    """Run DNA simulation and capture voltage traces."""
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
        
        # Create voltage history tracking
        voltage_history = np.zeros((len(NEURON_NAMES), TMAX), dtype=np.float32)
        spike_history = np.zeros((len(NEURON_NAMES), TMAX), dtype=np.uint8)
        
        # Import voltage arrays
        from src.neuron import _V, _VHIST, _SPIKES, _INPUT, _vpeak
        from src.network import convert_to_sparse_weights, sparse_matrix_multiply_only_spiking, _sparse_connections
        
        # Store initial voltage
        voltage_history[:, 0] = _V.copy()
        
        # Run simulation
        N = len(all_neurons)
        spikers = np.zeros(N, np.uint8)
        sparse_weights = convert_to_sparse_weights(dna_matrix)
        
        for t in range(TMAX - 1):
            # Apply synaptic input
            if spikers.any():
                post_I = sparse_matrix_multiply_only_spiking(spikers.astype(np.float32), 
                                                            sparse_weights, 
                                                            _sparse_connections)
                lend = min(alpha_array.size, TMAX - t - 1)
                _INPUT[:, t+1:t+1+lend] += post_I[:, None] * alpha_array[:lend]
            
            # Take integration step
            spikers = vectorised_step(_INPUT[:, t])
            spike_history[:, t] = spikers
            
            # Record voltage
            if t > 0:
                peaked = spike_history[:, t-1].astype(bool)
                voltage_history[peaked, t-1] = _vpeak[peaked]
            
            voltage_history[:, t+1] = _V.copy()
        
        # Handle final timestep
        final_spikers = vectorised_step(_INPUT[:, TMAX-1])
        spike_history[:, TMAX-1] = final_spikers
        
        # Create voltage traces dictionary
        voltage_traces = {}
        for i, neuron_name in enumerate(NEURON_NAMES):
            voltage_traces[neuron_name] = voltage_history[i, :]
        
        # Calculate spike counts
        spike_counts = evaluate_conditions(spike_history)
        
        results[label] = {
            'voltages': voltage_traces,
            'spike_counts': spike_counts[label] if label in spike_counts else 0,
            'spikes': spike_history.copy()
        }
        
        print(f"  Voltage range: {np.min(voltage_history):.2f} to {np.max(voltage_history):.2f} mV")
        print(f"  Score: {spike_counts[label] if label in spike_counts else 0}")
        
        # Show detailed scoring diagnostics
        print(f"  Detailed scoring failures for {label}:")
        diagnose_conditions(spike_history, label, 20)
    
    return results

def create_voltage_plot(results, weight_info):
    """Create interactive voltage trace plot."""
    time_ms = np.arange(TMAX)
    n_neurons = len(NEURON_NAMES)
    
    subplot_titles = []
    for neuron_name in NEURON_NAMES:
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
        
        # Add stimulus markers
        for col in [1, 2]:
            fig.add_vrect(
                x0=1000, x1=1200,
                fillcolor="red", opacity=0.2,
                layer="below", line_width=0,
                row=row, col=col
            )
            fig.add_vrect(
                x0=3000, x1=3100,
                fillcolor="green", opacity=0.2,
                layer="below", line_width=0,
                row=row, col=col
            )
    
    # Update layout
    fig.update_layout(
        title=f'Custom Weights Voltage Traces | Scores: Exp={results["experimental"]["spike_counts"]}, ' + 
              f'Ctrl={results["control"]["spike_counts"]}, Total={results["experimental"]["spike_counts"] + results["control"]["spike_counts"]} | ' +
              f'Connections: {weight_info["num_connections"]}' +
              f'<br><sub>* = Criteria neurons | Red=Cue (1000-1200ms) | Green=Go (3000-3100ms)</sub>',
        height=300 * n_neurons,
        width=1200,
        showlegend=False,
        hovermode='closest'
    )
    
    # Update y-axes with borders for criteria neurons
    for i in range(1, n_neurons + 1):
        neuron_name = NEURON_NAMES[i-1]
        is_criteria_neuron = neuron_name in CRITERIA_NAMES
        
        if is_criteria_neuron:
            border_style = dict(linewidth=3, linecolor='gold', mirror=True)
        else:
            border_style = dict(linewidth=1, linecolor='lightgray', mirror=True)
        
        fig.update_yaxes(range=[-100, 100], title_text="Voltage (mV)", row=i, col=1, **border_style)
        fig.update_yaxes(range=[-100, 100], title_text="Voltage (mV)", row=i, col=2, **border_style)
        fig.update_xaxes(row=i, col=1, **border_style)
        fig.update_xaxes(row=i, col=2, **border_style)
    
    # Update x-axes
    fig.update_xaxes(title_text="Time (ms)", row=n_neurons, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=n_neurons, col=2)
    
    return fig

def create_matplotlib_plot(results, weight_info):
    """Create matplotlib voltage trace plot."""
    time_ms = np.arange(TMAX)
    n_neurons = len(NEURON_NAMES)
    
    fig, axes = plt.subplots(n_neurons, 2, figsize=(16, 2*n_neurons))
    total_score = results["experimental"]["spike_counts"] + results["control"]["spike_counts"]
    fig.suptitle(f'Custom Weights Voltage Traces | Scores: Exp={results["experimental"]["spike_counts"]}, ' +
                f'Ctrl={results["control"]["spike_counts"]}, Total={total_score} | Connections: {weight_info["num_connections"]}\n' +
                f'* = Criteria neurons | Red=Cue | Green=Go', fontsize=14)
    
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
        ax_exp.set_ylim(-100, 100)
        ax_exp.grid(True, alpha=0.3)
        
        # Add border for criteria neurons
        if is_criteria_neuron:
            for spine in ax_exp.spines.values():
                spine.set_linewidth(3)
                spine.set_color('gold')
        
        # Add stimulus markers
        ax_exp.axvspan(1000, 1200, alpha=0.2, color='red', label='Cue' if i == 0 else '')
        ax_exp.axvspan(3000, 3100, alpha=0.2, color='green', label='Go' if i == 0 else '')
        
        # Control condition  
        ax_ctrl = axes[i, 1]
        voltages_ctrl = results['control']['voltages'][neuron_name]
        ax_ctrl.plot(time_ms, voltages_ctrl, 'r-', linewidth=line_width)
        ax_ctrl.set_title(f'{neuron_name}{title_marker} - Control')
        ax_ctrl.set_ylabel('Voltage (mV)')
        ax_ctrl.set_ylim(-100, 100)
        ax_ctrl.grid(True, alpha=0.3)
        
        # Add border for criteria neurons
        if is_criteria_neuron:
            for spine in ax_ctrl.spines.values():
                spine.set_linewidth(3)
                spine.set_color('gold')
        
        # Add stimulus markers
        ax_ctrl.axvspan(1000, 1200, alpha=0.2, color='red')
        ax_ctrl.axvspan(3000, 3100, alpha=0.2, color='green')
        
        # Only add x-label to bottom row
        if i == n_neurons - 1:
            ax_exp.set_xlabel('Time (ms)')
            ax_ctrl.set_xlabel('Time (ms)')
    
    # Add legend to first subplot
    axes[0, 0].legend(loc='upper right', fontsize='small')
    
    plt.tight_layout()
    return fig

def analyze_scoring_failures(results):
    """Detailed analysis of where points are being lost."""
    print(f"\n🔍 DETAILED SCORING ANALYSIS")
    print("="*80)
    
    # Get failure details for both conditions
    for condition in ['experimental', 'control']:
        failures = diagnose_conditions(results[condition]['spikes'], condition, None, return_list=True)
        
        print(f"\n📊 {condition.upper()} Condition Failures ({len(failures)} total):")
        print("-"*60)
        
        if not failures:
            print("  ✅ No failures - perfect score!")
            continue
            
        # Group failures by neuron
        by_neuron = {}
        for failure in failures:
            neuron = failure['neuron']
            if neuron not in by_neuron:
                by_neuron[neuron] = []
            by_neuron[neuron].append(failure)
        
        for neuron, neuron_failures in by_neuron.items():
            print(f"\n  🧠 {neuron}:")
            for failure in neuron_failures[:10]:  # Show first 10 failures per neuron
                time_range = f"{failure['t_start']:4d}-{failure['t_end']:4d}ms"
                wanted_str = "ON " if failure['wanted'] else "OFF"
                got_str = f"{failure['spikes']:3d} spikes" if failure['spikes'] else "silent"
                print(f"    • Bin {failure['period']:2d} [{time_range}]: wanted {wanted_str}, got {got_str}")
            
            if len(neuron_failures) > 10:
                print(f"    ... ({len(neuron_failures) - 10} more failures)")
        
        # Summary by time period
        print(f"\n  📈 Failure Timeline for {condition}:")
        time_bins = {}
        for failure in failures:
            period = failure['period']
            if period not in time_bins:
                time_bins[period] = 0
            time_bins[period] += 1
        
        # Show periods with most failures
        sorted_periods = sorted(time_bins.items(), key=lambda x: x[1], reverse=True)
        for period, count in sorted_periods[:10]:
            time_start = period * BIN_SIZE
            time_end = (period + 1) * BIN_SIZE
            print(f"    • Bin {period:2d} [{time_start:4d}-{time_end:4d}ms]: {count} neurons failing")

if __name__ == "__main__":
    # Use the new_jh_weights from constants.py
    print("🧬 Running Custom Weight Configuration")
    print("="*60)
    print("Using new_jh_weights from constants.py:")
    
    for i, (from_n, to_n, weight) in enumerate(new_jh_weights):
        inhibitory_marker = " (inhibitory)" if from_n in INHIBITORY_NEURONS else ""
        print(f"  {i+1:2d}. {from_n:10s} → {to_n:10s} | {weight:4d}{inhibitory_marker}")
    
    # Create DNA from the specific weights
    dna_vector, connections_found = create_dna_from_weights(new_jh_weights)
    
    weight_info = {
        'num_connections': len(connections_found),
        'total_requested': len(new_jh_weights),
        'connections': connections_found
    }
    
    print(f"\n🧬 DNA Vector created:")
    print(f"  Non-zero weights: {np.count_nonzero(dna_vector)}")
    print(f"  Total DNA length: {len(dna_vector)}")
    print(f"  DNA vector: {dna_vector}")
    
    # Show which neurons are criteria neurons
    print(f"\n⭐ CRITERIA_NAMES (neurons used for fitness evaluation):")
    for i, name in enumerate(CRITERIA_NAMES):
        print(f"  {i+1:2d}. {name}")
    
    # Run the simulation
    print(f"\n🔄 Running simulation...")
    results = run_dna_with_voltage_tracking(dna_vector)
    
    total_score = results["experimental"]["spike_counts"] + results["control"]["spike_counts"]
    print(f"\n📊 Results:")
    print(f"  Experimental score: {results['experimental']['spike_counts']}")
    print(f"  Control score: {results['control']['spike_counts']}")
    print(f"  TOTAL SCORE: {total_score}")
    
    # Analyze where points are being lost
    analyze_scoring_failures(results)
    
    # Create interactive plot
    print(f"\n📈 Creating interactive voltage trace plot...")
    fig_interactive = create_voltage_plot(results, weight_info)
    
    # Save interactive plot
    interactive_filename = 'custom_weights_voltage_traces_interactive.html'
    fig_interactive.write_html(interactive_filename)
    print(f"✅ Interactive plot saved to: {interactive_filename}")
    
    # Create static matplotlib plot
    print(f"\n📊 Creating static voltage trace plot...")
    fig_static = create_matplotlib_plot(results, weight_info)
    
    # Save static plot
    static_filename = 'custom_weights_voltage_traces.png'
    plt.savefig(static_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Static plot saved to: {static_filename}")
    
    # Save results
    results_data = {
        'dna_vector': dna_vector,
        'weight_connections': new_jh_weights,
        'connections_found': connections_found,
        'weight_info': weight_info,
        'simulation_results': results,
        'scores': {
            'experimental': results['experimental']['spike_counts'],
            'control': results['control']['spike_counts'],
            'total': total_score
        }
    }
    
    results_filename = 'custom_weights_results.pkl'
    import pickle
    with open(results_filename, 'wb') as f:
        pickle.dump(results_data, f)
    print(f"✅ Results data saved to: {results_filename}")
    
    print(f"\n🎉 ANALYSIS COMPLETE! 🎉")
    print("="*60)
    print("Key findings:")
    print(f"  • Used {len(connections_found)}/{len(new_jh_weights)} requested connections")
    print(f"  • Total fitness score: {total_score}")
    print(f"  • Breakdown: Experimental={results['experimental']['spike_counts']}, Control={results['control']['spike_counts']}")
    print(f"\nVisualization features:")
    print(f"  • Fixed y-axis (-100 to 100 mV) for all plots")
    print(f"  • Interactive hover functionality showing exact voltage values")
    print(f"  • GOLD BORDERS highlight neurons in CRITERIA_NAMES")
    print(f"  • Asterisk (*) markers in titles for criteria neurons")
    print(f"  • Red shading = Cue period (1000-1200ms)")
    print(f"  • Green shading = Go period (3000-3100ms)")
    print(f"\nTo view the interactive plot:")
    print(f"  Open '{interactive_filename}' in any web browser")
    print("="*60)
    
    plt.show()