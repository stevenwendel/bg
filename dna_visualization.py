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

def create_voltage_plot(results, dna_info):
    """Create interactive voltage trace plot for a single DNA with missed scoring highlights."""
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
        
        # Add missed scoring highlights for criteria neurons only
        if is_criteria_neuron:
            # Experimental missed points
            exp_missed = [mp for mp in results['experimental']['missed_points'] if mp['neuron'] == neuron_name]
            for missed in exp_missed:
                fig.add_vrect(
                    x0=missed['t_start'], x1=missed['t_end'],
                    fillcolor="orange", opacity=0.4,
                    layer="below", line_width=0,
                    row=row, col=1,
                    annotation_text=f"Miss: W{missed['wanted']} G{missed['spikes']}",
                    annotation_position="top left",
                    annotation_font_size=8
                )
            
            # Control missed points  
            ctrl_missed = [mp for mp in results['control']['missed_points'] if mp['neuron'] == neuron_name]
            for missed in ctrl_missed:
                fig.add_vrect(
                    x0=missed['t_start'], x1=missed['t_end'],
                    fillcolor="orange", opacity=0.4,
                    layer="below", line_width=0,
                    row=row, col=2,
                    annotation_text=f"Miss: W{missed['wanted']} G{missed['spikes']}",
                    annotation_position="top left",
                    annotation_font_size=8
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
    
    # Calculate total missed points for title
    exp_missed_total = len(results['experimental']['missed_points'])
    ctrl_missed_total = len(results['control']['missed_points'])
    total_missed = exp_missed_total + ctrl_missed_total
    
    # Handle different original_dna formats for title
    orig_dna = dna_info['original_dna']
    if isinstance(orig_dna, dict) and 'run_folder' in orig_dna:
        run_info = f'Run: {orig_dna["run_folder"]}'
    else:
        run_info = 'Run: Successful Vector'
    
    # Update layout
    fig.update_layout(
        title=f'DNA {dna_info["id"]} - Score: {dna_info["pruned_score"]} | ' + 
              f'Weights: {dna_info["original_nonzero"]}→{dna_info["pruned_nonzero"]} | ' +
              f'{run_info}' +
              f'<br><sub>* = Criteria neurons | Red=Cue | Green=Go | Orange=Missed Points ({total_missed} total: {exp_missed_total} exp + {ctrl_missed_total} ctrl)</sub>',
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