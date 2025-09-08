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
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
from src.constants import (
    NEURON_NAMES, CRITERIA_NAMES, INHIBITORY_NEURONS, 
    ACTIVE_SYNAPSES, TMAX
)

def _draw_curved_arrow(ax, pos1, pos2, weight, is_inhibitory, connectionstyle="arc3,rad=0.15"):
    """Draw a curved arrow between two positions."""
    color = 'red' if is_inhibitory else 'black'
    linestyle = '--' if is_inhibitory else '-'
    
    # Calculate node radius from node size (2500)
    node_radius = np.sqrt(2500 / np.pi) * 0.6  # Approximate radius with some padding
    
    arrow = FancyArrowPatch(
        pos1, pos2,
        arrowstyle='-|>',
        mutation_scale=15,   # Consistent arrow head size
        linewidth=2.0,       # Match main arrow width
        linestyle=linestyle,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=node_radius,  # Shrink to node edge
        shrinkB=node_radius,  # Shrink to node edge
        zorder=1
    )
    ax.add_patch(arrow)
    
    # Add weight label at midpoint of curve
    # Calculate approximate midpoint of curved path
    mid_x = (pos1[0] + pos2[0]) / 2
    mid_y = (pos1[1] + pos2[1]) / 2
    
    # Offset perpendicular to line for curved paths
    dx, dy = pos2[0] - pos1[0], pos2[1] - pos1[1]
    length = np.hypot(dx, dy)
    if length > 0:
        perp_x, perp_y = -dy/length, dx/length
        # Offset based on curvature direction
        radius = float(connectionstyle.split('rad=')[1].rstrip(')'))
        offset_factor = 0.3 * radius
        mid_x += perp_x * offset_factor * length
        mid_y += perp_y * offset_factor * length
    
    ax.text(mid_x, mid_y, f'{weight}',
            fontsize=7, color='gray',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'),
            zorder=3)

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
    fig = plot_neurons_interactive(
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
    
    # Set y-axis range to -100 to 100mV for all subplots
    fig.update_yaxes(range=[-100, 100])
    
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
    
    # Define custom positions for each node (from your improved layout)
    neu_coords = {
        'Somat': (5, 8.5),
        'MSN1': (5, 6),
        'MSN2': (3, 6),
        'MSN3': (0, 6),
        'SNR1': (5, 2.5),
        'SNR2': (3, 2.5),
        'SNR3': (0, 2.5),
        'ALMinter': (2, 8.5),
        'PPN': (2, 0),
        'THALgo': (2, 4.5),
        'VMprep': (4, 1),
        'ALMprep': (4, 7),
        'ALMresp': (1, 7),
        'VMresp': (1, 1)
    }
    
    return G, neu_coords, connections

def create_network_plot(G, neu_coords, connections, dna_info):
    """Create improved matplotlib network topology plot with proper arrows and layout."""
    
    # Separate connections by type for different rendering
    main_edges = []
    vm_edges = []
    
    # Process all connections
    for pre, post, weight in connections:
        # Check for special VM edges that need custom rendering
        if (pre == 'VMresp' and post == 'ALMresp') or (pre == 'VMprep' and post == 'ALMprep'):
            vm_edges.append((pre, post, weight))
        else:
            main_edges.append((pre, post))
    
    # Find reciprocal edges for better rendering
    reciprocal_pairs = []
    single_edges = []
    
    for i, (pre1, post1) in enumerate(main_edges):
        found_reciprocal = False
        for j, (pre2, post2) in enumerate(main_edges):
            if i != j and pre1 == post2 and post1 == pre2:
                if (pre1, post1) not in [pair[0] for pair in reciprocal_pairs] and \
                   (post1, pre1) not in [pair[0] for pair in reciprocal_pairs]:
                    reciprocal_pairs.append(((pre1, post1), (pre2, post2)))
                found_reciprocal = True
                break
        if not found_reciprocal and (pre1, post1) not in [edge for pair in reciprocal_pairs for edge in pair]:
            single_edges.append((pre1, post1))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1) Draw nodes and labels - make nodes slightly larger
    nx.draw_networkx_nodes(G, pos=neu_coords, node_size=2500, 
                          node_color='lightblue', ax=ax)
    nx.draw_networkx_labels(G, pos=neu_coords, font_size=10, 
                           font_weight='bold', ax=ax)
    
    # 2) Draw single edges with consistent width and proper arrow positioning
    if single_edges:
        excitatory_single = [edge for edge in single_edges if edge[0] not in INHIBITORY_NEURONS]
        inhibitory_single = [edge for edge in single_edges if edge[0] in INHIBITORY_NEURONS]
        
        # Calculate node radius for proper shrinking
        node_radius = np.sqrt(2500 / np.pi) * 0.6  # Approximate radius with padding
        
        # Draw single excitatory edges
        if excitatory_single:
            nx.draw_networkx_edges(
                G, pos=neu_coords, 
                edgelist=excitatory_single,
                width=2.0,  # Consistent width with curved arrows
                style='solid',
                edge_color='black',
                arrowsize=15,  # Consistent arrow size
                node_size=2500,  # Match node size for proper shrinking
                min_source_margin=node_radius,  # Don't overlap source node
                min_target_margin=node_radius,  # Don't overlap target node
                ax=ax
            )
        
        # Draw single inhibitory edges  
        if inhibitory_single:
            nx.draw_networkx_edges(
                G, pos=neu_coords, 
                edgelist=inhibitory_single,
                width=2.0,  # Consistent width with curved arrows
                style='dashed',
                edge_color='red',
                arrowsize=15,  # Consistent arrow size
                node_size=2500,  # Match node size for proper shrinking
                min_source_margin=node_radius,  # Don't overlap source node
                min_target_margin=node_radius,  # Don't overlap target node
                ax=ax
            )
    
    # 3) Draw reciprocal edges with offset curves
    if reciprocal_pairs:
        for edge_pair in reciprocal_pairs:
            (pre1, post1), (pre2, post2) = edge_pair
            
            # Get connection weights
            weight1 = next(w for p, po, w in connections if p == pre1 and po == post1)
            weight2 = next(w for p, po, w in connections if p == pre2 and po == post2)
            
            # Draw curved arrows for reciprocal connections
            pos1 = neu_coords[pre1]
            pos2 = neu_coords[post1]
            
            # Draw first direction with positive curvature
            _draw_curved_arrow(ax, pos1, pos2, weight1, 
                              pre1 in INHIBITORY_NEURONS, 
                              connectionstyle="arc3,rad=0.15")
            
            # Draw second direction with negative curvature  
            _draw_curved_arrow(ax, pos2, pos1, weight2,
                              pre2 in INHIBITORY_NEURONS,
                              connectionstyle="arc3,rad=-0.15")
    
    # 4) Draw VM-specific edges as compact, parallel arrows with consistent styling
    if vm_edges:
        offset_dist = 0.25  # push the two arrows apart
        head_size = 15      # Match other arrows
        shaft_width = 2.0   # Consistent width with other arrows
        # Calculate proper node radius for shrinkage
        node_radius = np.sqrt(2500 / np.pi) * 0.6  # Same as other arrows
        
        for idx, (u, v, weight) in enumerate(vm_edges):
            x1, y1 = neu_coords[u]
            x2, y2 = neu_coords[v]
            dx, dy = x2 - x1, y2 - y1
            length = np.hypot(dx, dy)
            
            # perpendicular unit-vector for left/right offset
            perp = np.array([-dy, dx]) / length if length else np.zeros(2)
            sign = 1 if idx % 2 == 0 else -1  # one edge above, the other below
            
            start = np.array([x1, y1]) + perp * offset_dist * sign
            end = np.array([x2, y2]) + perp * offset_dist * sign
            
            # Draw arrow with consistent styling
            arrow = FancyArrowPatch(
                posA=tuple(start), posB=tuple(end),
                arrowstyle='-|>',
                mutation_scale=head_size,
                linewidth=shaft_width,
                shrinkA=node_radius,  # Use calculated node radius
                shrinkB=node_radius,  # Use calculated node radius
                color='black',
                zorder=1
            )
            ax.add_patch(arrow)
            
            # Add weight label with better positioning
            mid = (start + end) / 2
            ax.text(
                mid[0], mid[1], f'{weight}',
                fontsize=7,
                color='gray',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'),
                zorder=3
            )
    
    # 5) Add edge labels for single edges only (reciprocal edges have their own labels)
    single_edge_labels = {}
    reciprocal_edge_tuples = [edge for pair in reciprocal_pairs for edge in pair]
    vm_edge_tuples = [(e[0], e[1]) for e in vm_edges]
    
    for pre, post, weight in connections:
        if (pre, post) in single_edges and (pre, post) not in vm_edge_tuples:
            single_edge_labels[(pre, post)] = f'{weight}'
    
    if single_edge_labels:
        nx.draw_networkx_edge_labels(
            G, pos=neu_coords,
            edge_labels=single_edge_labels,
            font_color='gray',
            font_size=7,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'),
            ax=ax
        )
    
    # Add title with DNA information
    title = f'Network Topology - DNA {dna_info.get("id", "N/A")}\n'
    title += f'Score: {dna_info.get("pruned_score", "N/A")} | '
    title += f'Connections: {len(connections)} | '
    title += f'Weights: {dna_info.get("pruned_nonzero", "N/A")}'
    
    ax.set_title(title)
    ax.axis('off')
    
    plt.tight_layout()
    return fig