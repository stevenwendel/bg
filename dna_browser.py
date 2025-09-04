#!/usr/bin/env python3
"""
DNA Browser Module

Handles interactive DNA browsing widgets and interface.
"""

import matplotlib.pyplot as plt
from ipywidgets import IntSlider, Dropdown, VBox, HBox, Output
from IPython.display import clear_output
from src.constants import INHIBITORY_NEURONS
from dna_visualization import create_voltage_plot, create_directed_graph_from_dna, create_network_plot

def create_dual_dna_browser(target_dnas, pruned_results, simulation_results, 
                           pruning_threshold, max_pruned_weights):
    """Create interactive DNA browser with both voltage plots and network graphs."""
    # Use target DNAs if available, otherwise fall back to all pruned results
    dnas_for_viz = target_dnas if target_dnas else pruned_results
    sims_for_viz = simulation_results
    
    if not dnas_for_viz or not sims_for_viz:
        print("❌ No data available for browsing")
        return
    
    # Create widgets
    dna_slider = IntSlider(
        value=0,
        min=0,
        max=len(dnas_for_viz) - 1,
        step=1,
        description='DNA #:',
        style={'description_width': 'initial'},
        continuous_update=False
    )
    
    # Sort options
    sort_dropdown = Dropdown(
        options=[
            ('By Score (High→Low)', 'score_desc'),
            ('By Score (Low→High)', 'score_asc'),
            ('By Weights Removed (Most→Least)', 'removed_desc'),
            ('By Weights Removed (Least→Most)', 'removed_asc'),
            ('By Final Weight Count (Least→Most)', 'final_weights_asc'),
            ('By Final Weight Count (Most→Least)', 'final_weights_desc'),
            ('By Original Order', 'original')
        ],
        value='score_desc',
        description='Sort by:',
        style={'description_width': 'initial'}
    )
    
    # Visualization type selector
    viz_dropdown = Dropdown(
        options=[
            ('Voltage Traces', 'voltage'),
            ('Network Graph', 'network'),
            ('Both', 'both')
        ],
        value='both',
        description='Show:',
        style={'description_width': 'initial'}
    )
    
    output = Output()
    
    # Store sorted indices
    sorted_indices = list(range(len(dnas_for_viz)))
    
    def sort_data(sort_by):
        nonlocal sorted_indices
        
        if sort_by == 'score_desc':
            sorted_indices = sorted(range(len(dnas_for_viz)), 
                                  key=lambda i: dnas_for_viz[i]['pruned_score'], reverse=True)
        elif sort_by == 'score_asc':
            sorted_indices = sorted(range(len(dnas_for_viz)), 
                                  key=lambda i: dnas_for_viz[i]['pruned_score'])
        elif sort_by == 'removed_desc':
            sorted_indices = sorted(range(len(dnas_for_viz)), 
                                  key=lambda i: dnas_for_viz[i]['weights_removed'], reverse=True)
        elif sort_by == 'removed_asc':
            sorted_indices = sorted(range(len(dnas_for_viz)), 
                                  key=lambda i: dnas_for_viz[i]['weights_removed'])
        elif sort_by == 'final_weights_asc':
            sorted_indices = sorted(range(len(dnas_for_viz)), 
                                  key=lambda i: dnas_for_viz[i]['pruned_nonzero'])
        elif sort_by == 'final_weights_desc':
            sorted_indices = sorted(range(len(dnas_for_viz)), 
                                  key=lambda i: dnas_for_viz[i]['pruned_nonzero'], reverse=True)
        else:  # original
            sorted_indices = list(range(len(dnas_for_viz)))
        
        # Reset slider
        dna_slider.value = 0
    
    def update_plot(dna_index, sort_by, viz_type):
        with output:
            clear_output(wait=True)
            
            # Get actual index after sorting
            actual_index = sorted_indices[dna_index]
            
            dna_info = dnas_for_viz[actual_index]
            sim_result = sims_for_viz[actual_index]
            
            if sim_result is None:
                print(f"❌ No simulation data available for DNA {actual_index + 1}")
                return
            
            # Show if this is a target DNA
            is_target = target_dnas and dna_info in target_dnas
            target_marker = " 🎯 TARGET" if is_target else ""
            
            # Display DNA information
            print(f"🧬 DNA {dna_index + 1} of {len(dnas_for_viz)} (Original Index: {actual_index + 1}){target_marker}")
            print(f"📊 Scores: Original={dna_info['original_score']}, Pruned={dna_info['pruned_score']} "
                  f"(Exp:{dna_info['final_exp_score']}, Cont:{dna_info['final_cont_score']})")
            print(f"⚖️  Weights: {dna_info['original_nonzero']} → {dna_info['pruned_nonzero']} "
                  f"({dna_info['weights_removed']} removed, {dna_info['weights_removed']/dna_info['original_nonzero']*100:.1f}% reduction)")
            
            # Handle different original_dna formats
            orig_dna = dna_info['original_dna']
            if isinstance(orig_dna, dict):
                if 'run_folder' in orig_dna:
                    run_folder = orig_dna['run_folder']
                    generation = orig_dna.get('generation', 'Unknown')
                elif 'original_dna_id' in orig_dna:
                    run_folder = "Successful Vector"
                    generation = orig_dna.get('generation', orig_dna.get('pass', 'Unknown'))
                else:
                    run_folder = "Unknown"
                    generation = "Unknown"
            else:
                run_folder = "Unknown"
                generation = "Unknown"
            
            print(f"📁 Source: {run_folder}, Gen/Pass: {generation}")
            
            if is_target:
                print(f"🎯 TARGET CRITERIA MET:")
                print(f"  Score >= {pruning_threshold}: {dna_info['pruned_score']} ✅")
                print(f"  Weights <= {max_pruned_weights}: {dna_info['pruned_nonzero']} ✅")
            
            print(f"\\n🧬 Pruned DNA Vector:")
            print(f"  {dna_info['pruned_dna']}")
            
            # Show network connections
            G, neu_coords, connections = create_directed_graph_from_dna(dna_info['pruned_dna'], dna_info)
            if connections:
                print(f"\\n🔗 Network Connections ({len(connections)} total):")
                for from_neuron, to_neuron, weight in connections:
                    is_inhibitory = from_neuron in INHIBITORY_NEURONS
                    conn_type = "Inhibitory" if is_inhibitory else "Excitatory"
                    print(f"  {from_neuron:8s} → {to_neuron:8s} | {weight:6d} | {conn_type}")
            else:
                print("\\n⚠️  No connections found in pruned DNA")
            
            # Create and display plots based on selection
            if viz_type in ['voltage', 'both']:
                print("\\n📈 Voltage Traces:")
                voltage_fig = create_voltage_plot(sim_result, dna_info)
                voltage_fig.show()
            
            if viz_type in ['network', 'both']:
                print("\\n🌐 Network Topology:")
                if connections:
                    network_fig = create_network_plot(G, neu_coords, connections, dna_info)
                    plt.show()
                else:
                    print("  ⚠️  Cannot create network graph: No connections to display")
    
    # Set up interactions
    def on_sort_change(change):
        sort_data(change['new'])
        update_plot(dna_slider.value, change['new'], viz_dropdown.value)
    
    def on_slider_change(change):
        update_plot(change['new'], sort_dropdown.value, viz_dropdown.value)
    
    def on_viz_change(change):
        update_plot(dna_slider.value, sort_dropdown.value, change['new'])
    
    sort_dropdown.observe(on_sort_change, names='value')
    dna_slider.observe(on_slider_change, names='value')
    viz_dropdown.observe(on_viz_change, names='value')
    
    # Initial sort
    sort_data(sort_dropdown.value)
    
    # Create layout
    controls = HBox([sort_dropdown, viz_dropdown, dna_slider])
    
    # Display initial plot
    update_plot(0, sort_dropdown.value, viz_dropdown.value)
    
    return VBox([controls, output])