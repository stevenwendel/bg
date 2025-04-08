import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

ACTIVE_SYNAPSES = [
    ["Somat", "ALMprep"], ["Somat", "ALMinter"], ["Somat", "ALMresp"], ["Somat", "MSN1"],
    ["MSN1", "SNR1"], ["MSN1", "SNR2"], ["MSN1", "SNR3"],
    ["MSN2", "SNR1"], ["MSN2", "SNR2"], ["MSN2", "SNR3"],
    ["MSN3", "SNR1"], ["MSN3", "SNR2"], ["MSN3", "SNR3"],
    ["SNR1", "VMprep"], ["SNR1", "VMresp"],
    ["SNR2", "VMprep"], ["SNR2", "VMresp"],
    ["SNR3", "VMprep"], ["SNR3", "VMresp"],
    ["VMprep", "ALMprep"], ["VMprep", "ALMinter"], ["VMprep", "ALMresp"],
    ["VMresp", "ALMprep"], ["VMresp", "ALMinter"], ["VMresp", "ALMresp"],
    ["ALMprep", "MSN2"],
    ["ALMinter", "MSN1"], ["ALMinter", "MSN2"], ["ALMinter", "MSN3"],
    ["ALMresp", "MSN1"], ["ALMresp", "MSN2"], ["ALMresp", "MSN3"],
    ["ALMprep", "VMprep"], ["ALMresp", "VMresp"],
    ["MSN1", "MSN2"], ["MSN1", "MSN3"],
    ["MSN2", "MSN1"], ["MSN2", "MSN3"],
    ["MSN3", "MSN1"], ["MSN3", "MSN2"],
    ["ALMprep", "ALMinter"], ["ALMprep", "ALMresp"],
    ["ALMinter", "ALMprep"], ["ALMinter", "ALMresp"],
    ["ALMresp", "ALMprep"], ["ALMresp", "ALMinter"],
    ["PPN", "THALgo"], ["THALgo", "ALMinter"], ["THALgo", "ALMresp"],
]


NEURON_NAMES = ["Somat", "MSN1", "SNR1", "VMprep", "ALMprep", "MSN2", "SNR2", "PPN", "THALgo", "ALMinter", "MSN3", "SNR3", "ALMresp",  "VMresp"]
INHIBITORY_NEURONS = ["SNR1","SNR2", "SNR3", "MSN1", "MSN2", "MSN3", "ALMinter"]


neu_coords = {
    'Somat': (5, 8.5),
    'MSN1': (5, 6),
    'MSN2': (3.2, 6),
    'MSN3': (0, 6),
    'SNR1': (5, 2.5),
    'SNR2': (3.2, 2.5),
    'SNR3': (0, 2.5),
    'ALMinter': (2, 8.5),
    'PPN': (2, 0),
    'THALgo': (2, 4.5),
    'VMprep': (4, 1),
    'ALMprep': (4, 7),
    'ALMresp': (1, 7),
    'VMresp': (1, 1)
}
proportions_strings = ['100%', 0, 0, '58%', '50%', 0, 0, 0, '100%', 0, 0, 0, '100%', 0, 0, 0, '100%', 0, '100%', '100%', 0, 0, 0, 0, '100%', '100%', 0, 0, 0, 0, 0, '100%', '100%', '100%', 0, 0, 0, 0, 0, 0, 0, 0, '100%', 0, 0, 0, '100%', '100%', '100%']

proportions = [100, 0, 0, 58, 50, 0, 0, 0, 100, 0, 0, 0, 100, 0, 0, 0, 100, 0, 100, 100, 0, 0, 0, 0, 100, 100, 0, 0, 0, 0, 0, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 100, 100, 100]
jh = [40, 0, 0, 220, -90, 0, 0, 0, -50, 0, 0, 0, -90, -10, 0, 0, -100, 0, -50, 70, 0, 0, 0, 0, 85, 320, 0, 0, 0, 0, 0, 320, 80, 90, 0, 0, 0, 0, 0, 0, 0, 0, -200, 0, 0, 0, 60, 55, 30]

# Create directed graph
G = nx.DiGraph()

# Add nodes with positions
for node, pos in neu_coords.items():
    G.add_node(node, pos=pos)

# Add edges with weights from jh where weight is not 0
for i, (source, target) in enumerate(ACTIVE_SYNAPSES):
    if proportions[i] != 0:
        G.add_edge(source, target, weight=f'{proportions_strings[i]}')


# Add edges with weights from jh where weight is not 0
for i, (source, target) in enumerate(ACTIVE_SYNAPSES):
    if jh[i] != 0:
        G.add_edge(source, target, weight=jh[i])

# Get position dictionary for drawing
pos = nx.get_node_attributes(G, 'pos')

# Create a new graph for proportion strings
G_prop = nx.DiGraph()
for i, (source, target) in enumerate(ACTIVE_SYNAPSES):
    if proportions[i] != 0:
        G_prop.add_edge(source, target, weight=proportions_strings[i])

# Create figure
plt.figure(figsize=(12, 8))

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                      node_size=3000, alpha=0.7)

# Draw node labels
nx.draw_networkx_labels(G, pos)

# Draw edges with offset for reciprocal connections
drawn_edges = set()
for (u, v, d) in G.edges(data=True):
    if (v, u) not in drawn_edges:  # Check if we haven't drawn this edge yet
        # Check if reciprocal edge exists
        is_reciprocal = G.has_edge(v, u)
        
        # Determine line style based on source neuron
        u_style = 'dashed' if u in INHIBITORY_NEURONS else 'solid'
        v_style = 'dashed' if v in INHIBITORY_NEURONS else 'solid'
        
        if is_reciprocal:
            # Calculate offset for parallel edges
            pos_u = np.array(pos[u])
            pos_v = np.array(pos[v])
            diff = pos_v - pos_u
            perp = np.array([-diff[1], diff[0]]) / np.linalg.norm(diff)
            offset = 0.065 * perp
            red_offset = 0.18 * perp

            # Draw original grey edges and red edges
            edge_graph = nx.DiGraph([(u, v)])
            nx.draw_networkx_edges(edge_graph, 
                                 {u: pos_u + offset, v: pos_v + offset},
                                 edge_color='grey',
                                 arrows=True,
                                 arrowsize=20,
                                 node_size=2600,
                                 width=2.0,
                                 style=u_style)
            
            nx.draw_networkx_edges(edge_graph, 
                                 {u: pos_u + red_offset, v: pos_v + red_offset},
                                 edge_color='white',
                                 alpha=0.3,
                                 arrows=True,
                                 arrowsize=20,
                                 node_size=2600,
                                 style=u_style)
            
            # Draw second set of edges
            edge_graph = nx.DiGraph([(v, u)])
            nx.draw_networkx_edges(edge_graph,
                                 {v: pos_v - offset, u: pos_u - offset},
                                 edge_color='grey',
                                 arrows=True,
                                 arrowsize=20,
                                 node_size=2600,
                                 width=2.0,
                                 style=v_style)
            
            nx.draw_networkx_edges(edge_graph,
                                 {v: pos_v - red_offset, u: pos_u - red_offset},
                                 edge_color='white',
                                 alpha=0.3,
                                 arrows=True,
                                 arrowsize=20,
                                 node_size=2600,
                                 style=v_style)
            
            # Add weight labels for both grey and red edges
            edge_labels1 = {(u, v): d['weight']}
            edge_labels2 = {(v, u): G[v][u]['weight']}
            nx.draw_networkx_edge_labels(edge_graph, 
                                       {u: pos_u + offset, v: pos_v + offset}, 
                                       edge_labels1,
                                       )
            nx.draw_networkx_edge_labels(edge_graph, 
                                       {v: pos_v - offset, u: pos_u - offset}, 
                                       edge_labels2,
                                       )
            
            # Add proportion labels for red edges
            if G_prop.has_edge(u, v):
                prop_labels1 = {(u, v): G_prop[u][v]['weight']}
                nx.draw_networkx_edge_labels(edge_graph,
                                           {u: pos_u + red_offset, v: pos_v + red_offset},
                                           prop_labels1,
                                           font_color='red',
                                           alpha=.5)
            if G_prop.has_edge(v, u):
                prop_labels2 = {(v, u): G_prop[v][u]['weight']}
                nx.draw_networkx_edge_labels(edge_graph,
                                           {v: pos_v - red_offset, u: pos_u - red_offset},
                                           prop_labels2,
                                           font_color='red',
                                           alpha=.5)
            
            drawn_edges.add((u, v))
            drawn_edges.add((v, u))
        
        else:
            # Draw regular edge for non-reciprocal connections
            edge_graph = nx.DiGraph([(u, v)])
            # Original grey edge
            nx.draw_networkx_edges(edge_graph, pos,
                                 edge_color='grey',
                                 arrows=True,
                                 arrowsize=20,
                                 node_size=3000,
                                 width=2.0,
                                 style=u_style)
            
            # Red edge with slight offset
            pos_u = np.array(pos[u])
            pos_v = np.array(pos[v])
            diff = pos_v - pos_u
            perp = np.array([-diff[1], diff[0]]) / np.linalg.norm(diff)
            red_offset = 0.2 * perp
            
            nx.draw_networkx_edges(edge_graph,
                                 {u: pos_u + red_offset, v: pos_v + red_offset},
                                 edge_color='white',
                                 alpha=0.3,
                                 arrows=True,
                                 arrowsize=20,
                                 node_size=2600,
                                 style=u_style)
            
            # Add weight labels for grey edges
            edge_labels = {(u, v): d['weight']}
            nx.draw_networkx_edge_labels(edge_graph, pos, edge_labels)
            
            # Add proportion labels for red edges
            if G_prop.has_edge(u, v):
                prop_labels = {(u, v): G_prop[u][v]['weight']}
                nx.draw_networkx_edge_labels(edge_graph,
                                           {u: pos_u + red_offset, v: pos_v + red_offset},
                                           prop_labels,
                                           font_color='red',
                                           alpha=.5)
            
            drawn_edges.add((u, v))

plt.title("Proportion of connectomes with given connection")
plt.axis('off')
plt.tight_layout()
plt.show()