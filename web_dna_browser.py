#!/usr/bin/env python3
"""
Web-based DNA Browser - Renders plots in web browser
Alternative to Jupyter widgets that opens in a separate browser tab
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from copy import deepcopy

# Import project modules
from src.constants import *
from src.neuron import *
from src.network import *
from src.validation import *
from src.genetic_algorithm import *

# Import analysis modules
from dna_analyzer import find_all_aggregated_results, extract_high_scoring_dnas
from dna_simulation import run_dna_with_voltage_tracking, generate_all_simulation_results
from dna_visualization import create_voltage_plot, create_directed_graph_from_dna, create_network_plot
from weight_pruning import prune_dna_vectors


def create_web_voltage_plot(dna_info, sim_results, dna_index, total_dnas, 
                           show_experimental=True, show_control=True, 
                           show_missed_scoring=True):
    """Create interactive voltage plot using Plotly"""
    
    if not sim_results:
        return go.Figure().add_annotation(
            text="No simulation results available",
            x=0.5, y=0.5, showarrow=False,
            xref="paper", yref="paper"
        )
    
    # Determine which conditions to show
    conditions_to_show = []
    if show_experimental and 'exp' in sim_results:
        conditions_to_show.append(('exp', 'Experimental', 'blue'))
    if show_control and 'cont' in sim_results:
        conditions_to_show.append(('cont', 'Control', 'red'))
    
    if not conditions_to_show:
        return go.Figure().add_annotation(
            text="No conditions selected",
            x=0.5, y=0.5, showarrow=False,
            xref="paper", yref="paper"
        )
    
    # Create subplots
    n_neurons = len(NEURON_NAMES)
    n_conditions = len(conditions_to_show)
    
    fig = make_subplots(
        rows=n_neurons * n_conditions, 
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"{neuron} ({cond})" + (" 🎯" if neuron in CRITERIA_NAMES else "") 
                       for cond, _, _ in conditions_to_show 
                       for neuron in NEURON_NAMES],
        vertical_spacing=0.02
    )
    
    # Plot each condition and neuron
    row_idx = 1
    for cond_key, cond_name, color in conditions_to_show:
        condition_data = sim_results[cond_key]
        time_array = condition_data['time']
        voltage_dict = condition_data['voltage']
        
        for neuron_name in NEURON_NAMES:
            if neuron_name in voltage_dict:
                voltage = voltage_dict[neuron_name]
                
                # Add voltage trace
                fig.add_trace(
                    go.Scatter(
                        x=time_array,
                        y=voltage,
                        mode='lines',
                        name=f"{neuron_name} ({cond_name})",
                        line=dict(color=color, width=1),
                        showlegend=(neuron_name == NEURON_NAMES[0])  # Only show legend for first neuron
                    ),
                    row=row_idx, col=1
                )
                
                # Add missed scoring highlights
                if show_missed_scoring and 'missed_scoring' in condition_data:
                    missed_scoring = condition_data['missed_scoring']
                    if neuron_name in missed_scoring:
                        missed_info = missed_scoring[neuron_name]
                        
                        for period in missed_info['missed_periods']:
                            start_time = period['start_time']
                            end_time = period['end_time']
                            
                            # Add orange highlight rectangle
                            fig.add_shape(
                                type="rect",
                                x0=start_time, x1=end_time,
                                y0=min(voltage), y1=max(voltage),
                                fillcolor="orange",
                                opacity=0.3,
                                line=dict(color="orange", width=1),
                                row=row_idx, col=1
                            )
                            
                            # Add annotation
                            annotation_text = f"Miss: W{period['wanted']} G{period['got']}"
                            fig.add_annotation(
                                x=start_time + (end_time - start_time)/2,
                                y=max(voltage),
                                text=annotation_text,
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1,
                                arrowcolor="darkorange",
                                font=dict(size=8, color="darkorange"),
                                bgcolor="white",
                                bordercolor="orange",
                                row=row_idx, col=1
                            )
            
            row_idx += 1
    
    # Add stimulus markers
    for i in range(1, row_idx):
        # Cue stimulus
        fig.add_vline(x=1000, line=dict(color="green", dash="dash", width=2), row=i, col=1)
        fig.add_vline(x=1200, line=dict(color="green", dash="dot", width=2), row=i, col=1)
        
        # Go signal
        fig.add_vline(x=3000, line=dict(color="purple", dash="dash", width=2), row=i, col=1)
        fig.add_vline(x=3100, line=dict(color="purple", dash="dot", width=2), row=i, col=1)
    
    # Update layout
    title_text = (f"DNA {dna_index + 1}/{total_dnas} - "
                 f"Score: {dna_info['pruned_score']} | "
                 f"Weights: {dna_info['pruned_nonzero']} | "
                 f"Reduction: {dna_info['weights_removed']/dna_info['original_nonzero']*100:.1f}%")
    
    if dna_info['pruned_score'] >= 975 and dna_info['pruned_nonzero'] <= 18:
        title_text = "🎯 " + title_text
    
    fig.update_layout(
        title=dict(text=title_text, x=0.5),
        height=400 * n_neurons * n_conditions,
        showlegend=True,
        xaxis_title="Time (ms)",
        hovermode='x unified'
    )
    
    return fig


def create_dropdown_plots(target_dnas, simulation_results, pruning_threshold=975, max_weights=18):
    """Create dropdown menu for DNA selection with web-based plots"""
    
    # Create initial plot
    initial_fig = create_web_voltage_plot(
        target_dnas[0], simulation_results[0], 0, len(target_dnas)
    )
    
    # Create dropdown menu options
    dropdown_buttons = []
    for i, dna_info in enumerate(target_dnas):
        is_target = (dna_info['pruned_score'] >= pruning_threshold and 
                    dna_info['pruned_nonzero'] <= max_weights)
        
        button_label = (f"DNA {i+1}: Score {dna_info['pruned_score']}, "
                       f"Weights {dna_info['pruned_nonzero']}")
        if is_target:
            button_label = "🎯 " + button_label
        
        # Create the figure for this DNA
        fig_data = create_web_voltage_plot(
            dna_info, simulation_results[i], i, len(target_dnas)
        ).data
        
        dropdown_buttons.append({
            'label': button_label,
            'method': 'restyle',
            'args': [{'y': [trace.y for trace in fig_data],
                     'x': [trace.x for trace in fig_data]}]
        })
    
    # Add dropdown to initial figure
    initial_fig.update_layout(
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.1,
                y=1.15,
                xanchor="left",
                yanchor="top"
            )
        ],
        annotations=[
            dict(
                text="Select DNA:",
                x=0.05, y=1.15,
                xref="paper", yref="paper",
                align="left", showarrow=False
            )
        ]
    )
    
    return initial_fig


def load_cleaned_results():
    """Load from the cleaned_results folder that the notebook uses"""
    results_folder = "cleaned_results"
    
    if not os.path.exists(results_folder):
        print(f"❌ Results folder '{results_folder}' not found.")
        return None, None
    
    # Configuration matching the notebook
    SCORE_THRESHOLD = 980
    PRUNING_THRESHOLD = 975
    MAX_PRUNED_WEIGHTS = 18
    MAX_DNAS_TO_PROCESS = 1000
    
    print(f"🔍 Loading data from {results_folder}...")
    
    try:
        # Use the same functions as the notebook
        aggregated_files = find_all_aggregated_results(results_folder)
        if not aggregated_files:
            print(f"❌ No aggregated results found in {results_folder}")
            return None, None
        
        high_scoring_dnas = extract_high_scoring_dnas(
            aggregated_files, 
            SCORE_THRESHOLD,
            remove_duplicates=True,
            unique_configs_only=True
        )
        
        if len(high_scoring_dnas) > MAX_DNAS_TO_PROCESS:
            print(f"⚠️  Found {len(high_scoring_dnas)} DNAs, limiting to top {MAX_DNAS_TO_PROCESS}")
            high_scoring_dnas = high_scoring_dnas[:MAX_DNAS_TO_PROCESS]
        
        print(f"✅ Found {len(high_scoring_dnas)} high-scoring DNAs")
        
        # Prune DNAs
        print("🔧 Pruning DNAs...")
        pruned_results, successful_vectors = prune_dna_vectors(
            high_scoring_dnas, PRUNING_THRESHOLD, 
            method="fast_greedy", score_tolerance=5
        )
        
        # Create target DNAs
        target_dnas = []
        
        # Add successful vectors that meet criteria
        for sv in successful_vectors:
            if sv['nonzero_weights'] <= MAX_PRUNED_WEIGHTS:
                target_dna = {
                    'original_dna': sv,
                    'pruned_dna': sv['dna'],
                    'original_score': sv['score'],
                    'pruned_score': sv['score'],
                    'original_nonzero': sv['nonzero_weights'],
                    'pruned_nonzero': sv['nonzero_weights'],
                    'weights_removed': 0,
                    'final_exp_score': sv['exp_score'],
                    'final_cont_score': sv['cont_score'],
                    'id': sv['original_dna_id']
                }
                target_dnas.append(target_dna)
        
        # Add pruned results that meet criteria
        for result in pruned_results:
            meets_score = result['pruned_score'] >= PRUNING_THRESHOLD
            meets_weight = result['pruned_nonzero'] <= MAX_PRUNED_WEIGHTS
            
            if meets_score and meets_weight:
                existing_ids = [td['id'] for td in target_dnas]
                if result['id'] not in existing_ids:
                    target_dnas.append(result)
        
        print(f"🎯 Found {len(target_dnas)} target DNAs meeting both criteria")
        
        if not target_dnas:
            print("❌ No target DNAs found. Using top pruned results instead...")
            target_dnas = sorted(pruned_results, key=lambda x: x['pruned_score'], reverse=True)[:10]
        
        # Generate simulation results
        print("🧮 Generating simulation results...")
        simulation_results = generate_all_simulation_results(target_dnas)
        
        return target_dnas, simulation_results
        
    except Exception as e:
        print(f"❌ Error loading results: {str(e)}")
        return None, None


def main():
    """Main function to launch the web-based DNA browser"""
    print("🌐 Starting Web-based DNA Browser...")
    print("This will open interactive plots in your web browser to avoid Jupyter widget issues.\n")
    
    # Load the data
    target_dnas, simulation_results = load_cleaned_results()
    
    if target_dnas is None:
        print("\n💡 Make sure the cleaned_results folder contains aggregated_results.pkl files.")
        return
    
    # Create the interactive plot
    print(f"🎛️ Creating interactive web plot with {len(target_dnas)} DNAs...")
    
    fig = create_dropdown_plots(target_dnas, simulation_results)
    
    # Save as HTML and open in browser
    output_file = "dna_browser.html"
    pyo.plot(fig, filename=output_file, auto_open=True)
    
    print(f"✅ DNA Browser opened in web browser!")
    print(f"📄 Saved as: {output_file}")
    print(f"\nFeatures:")
    print("• Dropdown menu to select different DNA solutions")
    print("• Interactive voltage traces with zoom/pan")
    print("• Missed scoring highlights in orange")
    print("• Stimulus markers (green=cue, purple=go signal)")
    print("• 🎯 indicates target DNAs meeting criteria")


if __name__ == "__main__":
    main()