#!/usr/bin/env python3
"""
DNA Visualization Module

Handles visualization creation for DNA analysis including voltage plots and network graphs.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from src.constants import (
    NEURON_NAMES, CRITERIA_NAMES, INHIBITORY_NEURONS, 
    ACTIVE_SYNAPSES, TMAX
)

def create_voltage_plot(results, dna_info, condition="experimental", show_missed_scoring=True):
    """Create interactive voltage trace plot using the WORKING plot_neurons_interactive function."""
    # Import the working plotting function
    from src.viz import plot_neurons_interactive
    from src.network import create_experiment
    
    # Get stimulus waves for plotting
    splits, input_waves, alpha_array = create_experiment()
    cue_wave, go_wave = input_waves
    
    # Prepare voltage data in the format expected by plot_neurons_interactive
    # Extract voltage data for the specified condition
    hist_Vs = []
    hist_us = []  # We'll use dummy u values since we don't track them
    
    for neuron_name in NEURON_NAMES:
        voltage_data = results[condition]['voltages'][neuron_name]
        hist_Vs.append(voltage_data)
        # Create dummy u values (recovery variable) - just zeros
        hist_us.append(np.zeros_like(voltage_data))
    
    # Convert to numpy arrays as expected by plot_neurons_interactive
    hist_Vs = np.array(hist_Vs)
    hist_us = np.array(hist_us)
    
    # Get spike raster for missed scoring analysis
    spike_raster = None
    if show_missed_scoring and 'spike_raster' in results[condition]:
        spike_raster = results[condition]['spike_raster']
    
    # Create title with DNA info
    orig_dna = dna_info['original_dna']
    if isinstance(orig_dna, dict) and 'run_folder' in orig_dna:
        run_info = f'{orig_dna["run_folder"]}'
    else:
        run_info = 'Successful Vector'
    
    # Add condition info to title
    exp_score = results.get('experimental', {}).get('score', 'N/A')
    cont_score = results.get('control', {}).get('score', 'N/A')
    
    title = (f'DNA {dna_info["id"]} ({condition.title()}) - '
             f'Score: {dna_info["pruned_score"]} (Exp:{exp_score}, Cont:{cont_score}) | '
             f'Weights: {dna_info["original_nonzero"]}→{dna_info["pruned_nonzero"]} | '
             f'Run: {run_info}')
    
    # Use the working plotting function with missed scoring visualization
    return plot_neurons_interactive(
        hist_Vs=hist_Vs, 
        hist_us=hist_us, 
        neuron_names=NEURON_NAMES, 
        sq_wave=cue_wave, 
        go_wave=go_wave, 
        show_u=False,
        title=title,
        spike_raster=spike_raster,
        condition=condition,
        show_missed_scoring=show_missed_scoring
    )

def create_directed_graph_from_dna(dna_vector, dna_info):
    """Create directed graph representation from DNA vector."""
    G = nx.DiGraph()
    
    # Add nodes for all neurons
    for neuron_name in NEURON_NAMES:
        G.add_node(neuron_name)
    
    # Add edges with weights from DNA
    connections = []
    for i, (pre, post) in enumerate(ACTIVE_SYNAPSES):
        weight = int(dna_vector[i])
        if weight != 0:
            G.add_edge(pre, post, weight=weight)
            connections.append((pre, post, weight))
    
    # Create simple coordinates for layout
    neu_coords = {}
    n_neurons = len(NEURON_NAMES)
    for i, neuron_name in enumerate(NEURON_NAMES):
        angle = 2 * np.pi * i / n_neurons
        neu_coords[neuron_name] = (np.cos(angle), np.sin(angle))
    
    return G, neu_coords, connections

def create_network_plot(G, neu_coords, connections, dna_info):
    """Create matplotlib network topology plot."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Draw all nodes
    pos = neu_coords
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=1000, alpha=0.8, ax=ax)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
    
    # Separate inhibitory and excitatory connections
    inhibitory_edges = []
    excitatory_edges = []
    
    for pre, post, weight in connections:
        if pre in INHIBITORY_NEURONS:
            inhibitory_edges.append((pre, post))
        else:
            excitatory_edges.append((pre, post))
    
    # Draw edges with different colors
    if excitatory_edges:
        nx.draw_networkx_edges(G, pos, edgelist=excitatory_edges, 
                              edge_color='green', alpha=0.6, ax=ax)
    if inhibitory_edges:
        nx.draw_networkx_edges(G, pos, edgelist=inhibitory_edges, 
                              edge_color='red', alpha=0.6, ax=ax)
    
    # Add title
    ax.set_title(f'Network Topology - DNA {dna_info["id"]}\n'
                f'Score: {dna_info["pruned_score"]} | '
                f'Connections: {len(connections)} | '
                f'Green=Excitatory, Red=Inhibitory')
    
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig