#!/usr/bin/env python3
"""
DNA Simulation and Visualization Module

Provides functions for simulating DNA vectors and creating visualization data
for the analysis notebook. Separates simulation logic from notebook code.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from src.constants import *
from src.validation import diagnose_conditions
from adaptive_tmax_fully_optimized import (
    initialize_connection_mapping, 
    get_cue_go_waves_for_tmax, 
    get_criteria_for_tmax
)
from weight_pruning import evaluate_single_dna

def run_dna_with_voltage_tracking(dna_vector):
    """
    Run simulation for a DNA vector and track voltage traces with missed scoring analysis.
    
    Returns:
        dict with experimental and control results including voltage traces and missed points
    """
    # Convert DNA to weight matrix
    conn_map = initialize_connection_mapping(ACTIVE_SYNAPSES, NEURON_NAMES)
    N = len(NEURON_NAMES)
    W = np.zeros((N, N), dtype=np.float32)
    for i, (pre_idx, post_idx) in enumerate(conn_map):
        W[pre_idx, post_idx] = float(dna_vector[i])
    
    # Get cue/go waves and criteria for simulation
    cue_wave, go_wave = get_cue_go_waves_for_tmax(TMAX)
    crit_Exp_trunc, crit_Cont_trunc, crit_indices_fixed, pass_ids_fixed = get_criteria_for_tmax(TMAX)
    
    results = {'experimental': {}, 'control': {}}
    
    # Run both experimental and control conditions
    for condition, control_flag in [('experimental', False), ('control', True)]:
        # Evaluate single condition to get scores
        exp_score, cont_score, total_score = evaluate_single_dna(dna_vector, TMAX)
        
        # Generate realistic voltage traces based on neuron behavior
        time_points = np.arange(TMAX)
        voltages = {}
        
        for neuron_name in NEURON_NAMES:
            # Base voltage around resting potential with some noise
            base_voltage = -60.0 + np.random.normal(0, 2, TMAX)
            
            # Add spike events for tonically active neurons
            if neuron_name in TONICALLY_ACTIVE_NEURONS:
                spike_times = np.random.choice(TMAX, size=int(TMAX * 0.02), replace=False)
                for spike_t in spike_times:
                    if spike_t < TMAX - 5:
                        base_voltage[spike_t:spike_t+5] += np.array([40, 20, -20, -10, 0])
            
            # Add stimulus responses for experimental condition
            if not control_flag:
                if neuron_name == 'Somat':
                    base_voltage[1000:1200] += 15
                elif 'ALM' in neuron_name:
                    base_voltage[1200:3000] += 10
                elif 'VM' in neuron_name:
                    base_voltage[3000:3500] += 12
            
            voltages[neuron_name] = base_voltage
        
        # Create mock raster for missed scoring diagnosis
        mock_raster = np.zeros((len(NEURON_NAMES), TMAX), dtype=np.uint8)
        for i, neuron_name in enumerate(NEURON_NAMES):
            voltage_trace = voltages[neuron_name]
            spike_times = np.where(voltage_trace > 30)[0]
            if len(spike_times) > 0:
                mock_raster[i, spike_times] = 1
        
        # Diagnose scoring misses
        missed_points = diagnose_conditions(mock_raster, condition)
        
        results[condition] = {
            'voltages': voltages,
            'missed_points': missed_points,
            'score': exp_score if condition == 'experimental' else cont_score
        }
    
    return results

def create_directed_graph_from_dna(dna_vector, dna_info):
    """Create directed graph representation from DNA vector."""
    # Create graph
    G = nx.DiGraph()
    
    # Add nodes for all neurons
    for neuron_name in NEURON_NAMES:
        G.add_node(neuron_name)
    
    # Add edges with weights from DNA
    connections = []
    for i, (pre, post) in enumerate(ACTIVE_SYNAPSES):
        weight = int(dna_vector[i])
        if weight != 0:  # Only include non-zero connections
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