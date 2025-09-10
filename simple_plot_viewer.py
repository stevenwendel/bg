#!/usr/bin/env python3
"""
Simple Plot Viewer - Extract plotting from Jupyter and show in separate window
Works with already loaded data from the notebook
"""

import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend initially
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Switch to interactive backend after imports
try:
    matplotlib.use('Qt5Agg')
except:
    try:
        matplotlib.use('TkAgg')
    except:
        matplotlib.use('MacOSX')

def create_simple_plot_function():
    """
    Creates a simple plotting function that can be called from the Jupyter notebook
    This avoids the widget update issues by creating completely new plot windows
    """
    
    def plot_dna_in_new_window(dna_info, sim_results, dna_index=0, total_dnas=1, 
                               show_exp=True, show_cont=True, show_missed=True):
        """
        Create voltage plot in new matplotlib window
        Call this function from Jupyter notebook with your loaded data
        """
        
        # Create new figure
        fig, axes = plt.subplots(figsize=(16, 10))
        fig.suptitle(f'DNA {dna_index + 1}/{total_dnas} - Voltage Traces', 
                     fontsize=16, fontweight='bold')
        
        if not sim_results:
            axes.text(0.5, 0.5, 'No simulation results available', 
                     ha='center', va='center', transform=axes.transAxes, fontsize=14)
            plt.show()
            return
        
        # Import required constants (these should be available in the notebook environment)
        from src.constants import NEURON_NAMES, CRITERIA_NAMES, TMAX
        
        # Plot conditions
        y_offset = 0
        colors = {'exp': 'blue', 'cont': 'red'}
        
        conditions_to_plot = []
        if show_exp and 'exp' in sim_results:
            conditions_to_plot.append('exp')
        if show_cont and 'cont' in sim_results:
            conditions_to_plot.append('cont')
        
        for cond_idx, condition in enumerate(conditions_to_plot):
            condition_data = sim_results[condition]
            time_array = condition_data['time']
            voltage_dict = condition_data['voltage']
            
            cond_offset = cond_idx * (len(NEURON_NAMES) + 1) * 20
            
            for neuron_idx, neuron_name in enumerate(NEURON_NAMES):
                if neuron_name in voltage_dict:
                    voltage = voltage_dict[neuron_name]
                    y_pos = cond_offset + neuron_idx * 20
                    
                    # Plot voltage trace
                    axes.plot(time_array, voltage + y_pos, 
                             color=colors[condition], alpha=0.7, linewidth=1,
                             label=f"{neuron_name} ({condition})" if neuron_idx == 0 else "")
                    
                    # Add neuron label
                    label_text = f"{neuron_name} ({condition})"
                    if neuron_name in CRITERIA_NAMES:
                        label_text = "🎯 " + label_text
                    
                    axes.text(-200, y_pos, label_text, va='center', ha='right', fontsize=8)
                    
                    # Add missed scoring highlights if available
                    if (show_missed and 'missed_scoring' in condition_data and 
                        neuron_name in condition_data['missed_scoring']):
                        
                        missed_info = condition_data['missed_scoring'][neuron_name]
                        for period in missed_info['missed_periods']:
                            start_time = period['start_time']
                            end_time = period['end_time']
                            
                            # Add orange highlight
                            axes.axvspan(start_time, end_time, 
                                       ymin=(y_pos - 10 + 100) / (axes.get_ylim()[1] + 100),
                                       ymax=(y_pos + 10 + 100) / (axes.get_ylim()[1] + 100),
                                       alpha=0.3, color='orange')
                            
                            # Add annotation
                            annotation_text = f"Miss: W{period['wanted']} G{period['got']}"
                            axes.annotate(annotation_text, 
                                        xy=(start_time + (end_time - start_time)/2, y_pos),
                                        xytext=(5, 5), textcoords='offset points',
                                        fontsize=6, color='darkorange', fontweight='bold',
                                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Add stimulus markers
        axes.axvline(x=1000, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Cue Start')
        axes.axvline(x=1200, color='green', linestyle=':', alpha=0.7, linewidth=2, label='Cue End')
        axes.axvline(x=3000, color='purple', linestyle='--', alpha=0.7, linewidth=2, label='Go Start')
        axes.axvline(x=3100, color='purple', linestyle=':', alpha=0.7, linewidth=2, label='Go End')
        
        # Format plot
        axes.set_xlim(-300, TMAX + 300)
        axes.set_xlabel('Time (ms)', fontsize=10)
        axes.set_ylabel('Voltage + Offset (mV)', fontsize=10)
        axes.set_yticks([])
        
        # Add title with DNA info
        title = (f"Score: {dna_info['pruned_score']} | "
                f"Weights: {dna_info['pruned_nonzero']} | "
                f"Reduction: {dna_info['weights_removed']/dna_info['original_nonzero']*100:.1f}%")
        
        if dna_info['pruned_score'] >= 975 and dna_info['pruned_nonzero'] <= 18:
            title = "🎯 " + title
            
        axes.set_title(title, fontsize=12, fontweight='bold')
        
        # Add legends
        if len(conditions_to_plot) > 1:
            axes.legend(loc='upper right')
        
        # Show in new window
        plt.tight_layout()
        plt.show()
        
        return fig
    
    return plot_dna_in_new_window


def create_notebook_helper_functions():
    """
    Create helper functions that can be used directly in the Jupyter notebook
    """
    
    # Create the plotting function
    plot_dna_in_new_window = create_simple_plot_function()
    
    def browse_dnas_externally(target_dnas, simulation_results, start_index=0):
        """
        Simple function to browse through DNAs by calling it multiple times
        Usage in notebook:
            browse_dnas_externally(target_dnas, simulation_results, 0)  # Show DNA 1
            browse_dnas_externally(target_dnas, simulation_results, 1)  # Show DNA 2
            etc.
        """
        if start_index >= len(target_dnas):
            print(f"Index {start_index} is out of range. Max index: {len(target_dnas)-1}")
            return
        
        dna_info = target_dnas[start_index]
        sim_results = simulation_results[start_index]
        
        print(f"Displaying DNA {start_index + 1}/{len(target_dnas)} in new window...")
        plot_dna_in_new_window(dna_info, sim_results, start_index, len(target_dnas))
    
    def show_dna_network(target_dnas, dna_index=0):
        """
        Show network graph for a specific DNA in new window
        """
        if dna_index >= len(target_dnas):
            print(f"Index {dna_index} is out of range. Max index: {len(target_dnas)-1}")
            return
        
        dna_info = target_dnas[dna_index]
        dna_vector = dna_info['pruned_dna']
        
        # Import visualization functions
        from dna_visualization import create_directed_graph_from_dna, create_network_plot
        
        # Create network plot in new window
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle(f'Network Graph - DNA {dna_index + 1}', fontsize=14, fontweight='bold')
        
        try:
            G = create_directed_graph_from_dna(dna_vector)
            create_network_plot(G, ax)
            
            # Add info text
            info_text = (f"Score: {dna_info['pruned_score']}\n"
                        f"Weights: {dna_info['pruned_nonzero']}\n"
                        f"Reduction: {dna_info['weights_removed']/dna_info['original_nonzero']*100:.1f}%")
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating network graph:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.show()
        return fig
    
    return plot_dna_in_new_window, browse_dnas_externally, show_dna_network


# Print instructions when this module is imported
print("""
🎛️ DNA Browser Helper Functions Loaded!

To use these functions in your Jupyter notebook:

1. Import this module:
   from simple_plot_viewer import create_notebook_helper_functions
   plot_dna, browse_dnas, show_network = create_notebook_helper_functions()

2. Browse DNAs in separate windows:
   browse_dnas(target_dnas, simulation_results, 0)  # Show DNA 1
   browse_dnas(target_dnas, simulation_results, 1)  # Show DNA 2
   
3. Show individual DNA voltage plot:
   plot_dna(target_dnas[0], simulation_results[0], 0, len(target_dnas))
   
4. Show network graph:
   show_network(target_dnas, 0)  # Show network for DNA 1

These plots will open in separate windows and update properly!
""")

if __name__ == "__main__":
    print("This module provides helper functions for Jupyter notebooks.")
    print("Import it in your notebook to use the plotting functions.")