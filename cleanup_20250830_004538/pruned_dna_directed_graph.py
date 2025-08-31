#!/usr/bin/env python3

import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

# Import constants
from src.constants import ACTIVE_SYNAPSES, NEURON_NAMES, INHIBITORY_NEURONS

def create_pruned_dna_graph():
    """Create a directed graph visualization of the pruned DNA connections."""
    
    # Load pruned DNA
    with open('greedy_pruned_dna.pkl', 'rb') as f:
        data = pickle.load(f)
    
    pruned_dna = data['pruned_dna']
    
    # Print the DNA vector used
    print("Pruned DNA vector:")
    print(pruned_dna)
    print(f"Shape: {pruned_dna.shape}")
    print(f"Non-zero elements: {np.count_nonzero(pruned_dna)}")
    print()
    
    # Define custom positions for each node (from current_workbook.ipynb)
    neu_coords = {
        'Somat': (5, 8.5),
        'MSN1': (5, 5.5),
        'MSN2': (3, 5.5),
        'MSN3': (0, 5.5),
        'SNR1': (5, 2.5),
        'SNR2': (3, 2.5),
        'SNR3': (0, 2.5),
        'ALMinter': (2, 8.5),
        'PPN': (2.5, 0),
        'THALgo': (2, 4.5),
        'VMprep': (4, 1),
        'ALMprep': (4, 7),
        'ALMresp': (1, 7),
        'VMresp': (1, 1)
    }
    
    # Create directed graph and add all neuron nodes
    G = nx.DiGraph()
    G.add_nodes_from(NEURON_NAMES)
    
    # Extract non-zero connections from pruned DNA
    connections = []
    print("Pruned DNA connections:")
    print("From -> To | Weight | Type")
    print("-" * 35)
    
    for i, weight in enumerate(pruned_dna):
        if weight != 0:
            if i < len(ACTIVE_SYNAPSES):
                from_neuron, to_neuron = ACTIVE_SYNAPSES[i]
                
                # Determine connection type
                is_inhibitory = from_neuron in INHIBITORY_NEURONS
                style = 'dashed' if is_inhibitory else 'solid'
                
                # Determine edge width based on absolute weight
                abs_weight = abs(int(weight))
                if abs_weight >= 300:
                    width = 4
                elif abs_weight >= 100:
                    width = 3
                elif abs_weight >= 50:
                    width = 2
                else:
                    width = 1
                
                # Weight label with sign
                weight_label = f'{weight}'
                
                # Add edge with attributes
                G.add_edge(from_neuron, to_neuron, 
                          weight=weight_label, 
                          style=style, 
                          width=width,
                          raw_weight=int(weight))
                
                connections.append((from_neuron, to_neuron, int(weight)))
                
                # Print connection info
                conn_type = "Inhibitory" if is_inhibitory else "Excitatory"
                print(f"{from_neuron:8s} -> {to_neuron:8s} | {weight:6d} | {conn_type}")
    
    print(f"\nTotal connections: {len(connections)}")
    print(f"Original DNA had {data['original_nonzero']} non-zero weights")
    print(f"Pruned DNA has {data['final_nonzero']} non-zero weights")
    print(f"Reduction: {data['weights_removed']} weights removed ({data['weights_removed']/data['original_nonzero']*100:.1f}%)")
    
    return G, neu_coords, connections, data

def plot_pruned_network_graph(G, neu_coords, connections, pruned_info):
    """Plot the pruned network as a directed graph with custom styling."""
    
    # Create figure - more square aspect ratio
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    
    # Separate edges into reciprocal and non-reciprocal
    reciprocal_edges_pairs = set()
    non_reciprocal_edges_data = []
    all_edges_with_data = list(G.edges(data=True))
    
    # Check for reciprocal edges
    for u, v, data in all_edges_with_data:
        reverse_edge = (v, u)
        if G.has_edge(v, u):  # Reciprocal edge exists
            # Add to reciprocal set (use sorted tuple to avoid duplicates)
            reciprocal_edges_pairs.add(tuple(sorted([u, v])))
        else:  # Non-reciprocal edge
            non_reciprocal_edges_data.append((u, v, data))
    
    print(f"\nEdge analysis:")
    print(f"Reciprocal edge pairs: {len(reciprocal_edges_pairs)}")
    print(f"Non-reciprocal edges: {len(non_reciprocal_edges_data)}")
    
    # Draw nodes with uniform color
    nx.draw_networkx_nodes(G, pos=neu_coords, 
                          node_size=3000, 
                          node_color='lightblue', 
                          edgecolors='black',
                          linewidths=2,
                          ax=ax)
    
    # Draw node labels with better formatting
    nx.draw_networkx_labels(G, pos=neu_coords, 
                           font_size=10, 
                           font_weight='bold', 
                           font_color='black',
                           ax=ax)
    
    # Draw non-reciprocal edges
    if non_reciprocal_edges_data:
        print(f"Drawing {len(non_reciprocal_edges_data)} non-reciprocal edges...")
        edgelist_nr = [(u, v) for u, v, data in non_reciprocal_edges_data]
        styles_nr = [data['style'] for u, v, data in non_reciprocal_edges_data]
        widths_nr = [data['width'] for u, v, data in non_reciprocal_edges_data]
        
        nx.draw_networkx_edges(G, pos=neu_coords, 
                              edgelist=edgelist_nr,
                              node_size=3000,
                              arrowstyle='-|>', 
                              arrowsize=20,
                              edge_color='black',
                              width=widths_nr, 
                              style=styles_nr,
                              connectionstyle='arc3,rad=0', 
                              ax=ax)
    
    # Draw reciprocal edges manually with offset
    if reciprocal_edges_pairs:
        print(f"Drawing {len(reciprocal_edges_pairs)} reciprocal edge pairs...")
        
        parallel_offset = 0.1
        shorten_length = 0.35
        
        for u_orig, v_orig in reciprocal_edges_pairs:
            pos_u = np.array(neu_coords[u_orig])
            pos_v = np.array(neu_coords[v_orig])
            
            vec = pos_v - pos_u
            vec_len = np.linalg.norm(vec)
            if vec_len < 1e-6:
                continue
            
            unit_vec = vec / vec_len
            perp_vec = np.array([-unit_vec[1], unit_vec[0]])
            
            # Shorten to avoid overlap with nodes
            start_u = pos_u + unit_vec * shorten_length
            end_v = pos_v - unit_vec * shorten_length
            start_v = pos_v + unit_vec * shorten_length  
            end_u = pos_u - unit_vec * shorten_length
            
            # Draw both directions
            for direction, (start_pos, end_pos, src, dst) in enumerate([
                (start_u + perp_vec * parallel_offset, end_v + perp_vec * parallel_offset, u_orig, v_orig),
                (start_v - perp_vec * parallel_offset, end_u - perp_vec * parallel_offset, v_orig, u_orig)
            ]):
                # Get edge data
                edge_data = G[src][dst]
                edge_style = edge_data['style']
                edge_width = edge_data['width']
                
                arrow = FancyArrowPatch(start_pos, end_pos,
                                       arrowstyle='-|>',
                                       shrinkA=0, shrinkB=0,
                                       mutation_scale=20,
                                       linewidth=edge_width,
                                       linestyle='--' if edge_style == 'dashed' else '-',
                                       color='black',
                                       alpha=0.8)
                ax.add_patch(arrow)
    
    # Add edge labels for non-reciprocal edges
    if non_reciprocal_edges_data:
        labels_nr = {}
        all_edge_labels = nx.get_edge_attributes(G, 'weight')
        
        for u, v, data in non_reciprocal_edges_data:
            edge_tuple = (u, v)
            if edge_tuple in all_edge_labels:
                labels_nr[edge_tuple] = all_edge_labels[edge_tuple]
        
        if labels_nr:
            nx.draw_networkx_edge_labels(G, pos=neu_coords, 
                                        edge_labels=labels_nr,
                                        label_pos=0.5, 
                                        rotate=True,
                                        font_color='black', 
                                        font_size=8, 
                                        bbox=dict(boxstyle='round,pad=0.2', 
                                                facecolor='white', 
                                                alpha=0.8),
                                        ax=ax)
    
    # Add edge labels for reciprocal edges manually
    if reciprocal_edges_pairs:
        label_perp_offset = parallel_offset * 1.5
        
        for u_orig, v_orig in reciprocal_edges_pairs:
            pos_u = np.array(neu_coords[u_orig])
            pos_v = np.array(neu_coords[v_orig])
            vec = pos_v - pos_u
            vec_len = np.linalg.norm(vec)
            if vec_len < 1e-6:
                continue
            
            unit_vec = vec / vec_len
            perp_vec = np.array([-unit_vec[1], unit_vec[0]])
            
            # Label positions
            mid_point = (pos_u + pos_v) / 2
            
            # Label for u -> v
            if G.has_edge(u_orig, v_orig):
                label_pos_uv = mid_point + perp_vec * label_perp_offset
                weight_uv = G[u_orig][v_orig]['weight']
                ax.text(label_pos_uv[0], label_pos_uv[1], weight_uv,
                       fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # Label for v -> u
            if G.has_edge(v_orig, u_orig):
                label_pos_vu = mid_point - perp_vec * label_perp_offset
                weight_vu = G[v_orig][u_orig]['weight']
                ax.text(label_pos_vu[0], label_pos_vu[1], weight_vu,
                       fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Set title and formatting
    plt.title(f'Pruned DNA Network Topology\n'
             f'Score: {pruned_info["final_scores"]["total"]} | '
             f'Connections: {pruned_info["final_nonzero"]}/{pruned_info["original_nonzero"]} | '
             f'Reduction: {pruned_info["weights_removed"]/pruned_info["original_nonzero"]*100:.1f}%\n'
             f'Dashed lines = connections from inhibitory neurons',
             fontsize=14, fontweight='bold', pad=20)
    
    # Set axis properties
    ax.set_aspect('equal')
    plt.grid(True, alpha=0.3)
    plt.xlabel('X Position', fontsize=12)
    plt.ylabel('Y Position', fontsize=12)
    
    # Adjust layout and show
    plt.tight_layout()
    
    return plt.gcf()

def main():
    """Main function to create and display the pruned DNA network graph."""
    
    print("🧬 Creating Pruned DNA Network Graph")
    print("=" * 50)
    
    # Create graph
    G, neu_coords, connections, pruned_info = create_pruned_dna_graph()
    
    # Plot graph
    fig = plot_pruned_network_graph(G, neu_coords, connections, pruned_info)
    
    # Save the plot
    output_file = '/Users/stevenwendel/Documents/GitHub/bg/pruned_dna_network_graph.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n💾 Network graph saved to: {output_file}")
    
    # Save the graph data
    graph_data = {
        'graph': G,
        'coordinates': neu_coords,
        'connections': connections,
        'pruned_info': pruned_info
    }
    
    with open('/Users/stevenwendel/Documents/GitHub/bg/pruned_dna_network_data.pkl', 'wb') as f:
        pickle.dump(graph_data, f)
    
    print("📊 Graph data saved to: pruned_dna_network_data.pkl")
    
    # Display the plot (commented out for non-interactive mode)
    # plt.show()
    
    print("\n🎉 Pruned DNA Network Analysis Complete!")
    print("Key insights:")
    print(f"  • Only {pruned_info['final_nonzero']} out of {pruned_info['original_nonzero']} connections needed")
    print(f"  • {pruned_info['weights_removed']/pruned_info['original_nonzero']*100:.1f}% weight reduction")
    print(f"  • Maintained fitness score of {pruned_info['final_scores']['total']}")
    print(f"  • Network shows sparse but effective connectivity")

if __name__ == "__main__":
    main()