#!/usr/bin/env python3
"""
Standalone DNA Browser - Renders plots in separate windows
Fixes Jupyter notebook widget update issues by using matplotlib with Qt backend
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.patches as mpatches
from pathlib import Path
import time
from copy import deepcopy

# Use the default macOS backend for better interactivity in separate windows
plt.switch_backend('macOSX')

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

class StandaloneDNABrowser:
    def __init__(self, target_dnas, simulation_results, pruning_threshold=975, max_weights=18):
        self.target_dnas = target_dnas
        self.simulation_results = simulation_results
        self.pruning_threshold = pruning_threshold
        self.max_weights = max_weights
        
        # Current selection
        self.current_dna_idx = 0
        self.show_experimental = True
        self.show_control = True
        self.show_missed_scoring = True
        
        # Create the main window
        self.create_interface()
    
    def create_interface(self):
        """Create the main interface with voltage plots and controls"""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('DNA Browser - Standalone Window', fontsize=16, fontweight='bold')
        
        # Create subplot layout - main plot area and controls
        gs = self.fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3,
                                   left=0.05, right=0.95, top=0.92, bottom=0.15)
        
        # Main plot area (voltage traces)
        self.plot_ax = self.fig.add_subplot(gs[:3, :])
        
        # Control area at bottom
        self.control_ax = self.fig.add_subplot(gs[3, :])
        self.control_ax.set_xlim(0, 10)
        self.control_ax.set_ylim(0, 1)
        self.control_ax.axis('off')
        
        # DNA slider
        slider_ax = plt.axes([0.1, 0.05, 0.6, 0.03])
        self.dna_slider = Slider(slider_ax, 'DNA', 0, max(0, len(self.target_dnas)-1), 
                                valinit=0, valfmt='%d', valstep=1)
        self.dna_slider.on_changed(self.update_dna)
        
        # Condition buttons
        condition_ax = plt.axes([0.75, 0.02, 0.12, 0.08])
        self.condition_radio = RadioButtons(condition_ax, ('Both', 'Exp', 'Control'))
        self.condition_radio.on_clicked(self.update_condition)
        
        # Network graph button
        network_ax = plt.axes([0.88, 0.05, 0.08, 0.03])
        self.network_button = Button(network_ax, 'Network')
        self.network_button.on_clicked(self.show_network_graph)
        
        # Initial plot
        self.update_plot()
        
        # Show the window
        plt.show()
    
    def update_dna(self, val):
        """Update plot when DNA slider changes"""
        self.current_dna_idx = int(self.dna_slider.val)
        self.update_plot()
    
    def update_condition(self, label):
        """Update plot when condition selection changes"""
        if label == 'Both':
            self.show_experimental = True
            self.show_control = True
        elif label == 'Exp':
            self.show_experimental = True
            self.show_control = False
        elif label == 'Control':
            self.show_experimental = False
            self.show_control = True
        self.update_plot()
    
    def show_network_graph(self, event):
        """Open network graph in separate window"""
        if self.current_dna_idx < len(self.target_dnas):
            current_dna = self.target_dnas[self.current_dna_idx]
            dna_vector = current_dna['pruned_dna']
            
            # Create network graph in new window
            network_fig, network_ax = plt.subplots(figsize=(12, 8))
            network_fig.suptitle(f'Network Graph - DNA {self.current_dna_idx + 1}', fontsize=14, fontweight='bold')
            
            try:
                # Use the visualization function to create network plot
                G = create_directed_graph_from_dna(dna_vector)
                create_network_plot(G, network_ax)
                
                # Add info text
                info_text = (f"Score: {current_dna['pruned_score']}\n"
                           f"Weights: {current_dna['pruned_nonzero']}\n"
                           f"Reduction: {current_dna['weights_removed']/current_dna['original_nonzero']*100:.1f}%")
                network_ax.text(0.02, 0.98, info_text, transform=network_ax.transAxes, 
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
            except Exception as e:
                network_ax.text(0.5, 0.5, f'Error creating network graph:\n{str(e)}', 
                               ha='center', va='center', transform=network_ax.transAxes)
            
            plt.show()
    
    def update_plot(self):
        """Update the voltage plot with current DNA and conditions"""
        self.plot_ax.clear()
        
        if self.current_dna_idx >= len(self.target_dnas):
            self.plot_ax.text(0.5, 0.5, 'No DNA selected', ha='center', va='center', 
                             transform=self.plot_ax.transAxes, fontsize=14)
            self.fig.canvas.draw()
            return
        
        # Get current DNA and simulation results
        current_dna = self.target_dnas[self.current_dna_idx]
        current_sim = self.simulation_results[self.current_dna_idx]
        
        try:
            # Create voltage plot
            self.create_voltage_plot(current_dna, current_sim)
            
        except Exception as e:
            self.plot_ax.text(0.5, 0.5, f'Error creating plot:\n{str(e)}', 
                             ha='center', va='center', transform=self.plot_ax.transAxes, fontsize=12)
        
        # Update title with DNA info
        title = (f"DNA {self.current_dna_idx + 1}/{len(self.target_dnas)} - "
                f"Score: {current_dna['pruned_score']} | "
                f"Weights: {current_dna['pruned_nonzero']} | "
                f"Reduction: {current_dna['weights_removed']/current_dna['original_nonzero']*100:.1f}%")
        
        if current_dna['pruned_score'] >= self.pruning_threshold and current_dna['pruned_nonzero'] <= self.max_weights:
            title = "🎯 " + title
            
        self.plot_ax.set_title(title, fontsize=12, fontweight='bold')
        
        self.fig.canvas.draw()
    
    def create_voltage_plot(self, dna_info, sim_results):
        """Create voltage traces with missed scoring visualization"""
        if not sim_results:
            self.plot_ax.text(0.5, 0.5, 'No simulation results available', 
                             ha='center', va='center', transform=self.plot_ax.transAxes)
            return
        
        # Determine which conditions to show
        conditions_to_show = []
        if self.show_experimental and 'exp' in sim_results:
            conditions_to_show.append(('exp', 'Experimental', 'blue'))
        if self.show_control and 'cont' in sim_results:
            conditions_to_show.append(('cont', 'Control', 'red'))
        
        if not conditions_to_show:
            self.plot_ax.text(0.5, 0.5, 'No conditions selected', 
                             ha='center', va='center', transform=self.plot_ax.transAxes)
            return
        
        # Plot each condition
        for i, (cond_key, cond_name, color) in enumerate(conditions_to_show):
            condition_data = sim_results[cond_key]
            
            # Plot voltage traces for each neuron type
            time_array = condition_data['time']
            voltage_dict = condition_data['voltage']
            
            y_offset = i * (len(NEURON_NAMES) + 1) * 20  # Separate conditions vertically
            
            for j, neuron_name in enumerate(NEURON_NAMES):
                if neuron_name in voltage_dict:
                    voltage = voltage_dict[neuron_name]
                    y_pos = y_offset + j * 20
                    
                    # Plot voltage trace
                    self.plot_ax.plot(time_array, voltage + y_pos, color=color, alpha=0.7, linewidth=1)
                    
                    # Add neuron label
                    label_text = f"{neuron_name} ({cond_name})"
                    if neuron_name in CRITERIA_NAMES:
                        label_text = "🎯 " + label_text
                    
                    self.plot_ax.text(-200, y_pos, label_text, va='center', ha='right', fontsize=8)
            
            # Add missed scoring visualization if enabled
            if self.show_missed_scoring and 'missed_scoring' in condition_data:
                self.add_missed_scoring_highlights(condition_data['missed_scoring'], 
                                                  y_offset, color, cond_name)
        
        # Add stimulus markers
        self.add_stimulus_markers()
        
        # Set plot limits and labels
        self.plot_ax.set_xlim(-300, TMAX + 300)
        self.plot_ax.set_xlabel('Time (ms)', fontsize=10)
        self.plot_ax.set_ylabel('Voltage + Offset (mV)', fontsize=10)
        
        # Remove y-ticks for cleaner look
        self.plot_ax.set_yticks([])
        
        # Add legend for conditions
        if len(conditions_to_show) > 1:
            legend_elements = [plt.Line2D([0], [0], color=color, label=name) 
                              for _, name, color in conditions_to_show]
            self.plot_ax.legend(handles=legend_elements, loc='upper right')
    
    def add_missed_scoring_highlights(self, missed_scoring, y_offset, color, condition_name):
        """Add orange highlights for missed scoring periods"""
        if not missed_scoring:
            return
        
        for neuron_idx, neuron_name in enumerate(NEURON_NAMES):
            if neuron_name in missed_scoring:
                missed_info = missed_scoring[neuron_name]
                y_pos = y_offset + neuron_idx * 20
                
                for period in missed_info['missed_periods']:
                    start_time = period['start_time']
                    end_time = period['end_time']
                    
                    # Add orange highlight rectangle
                    rect = mpatches.Rectangle((start_time, y_pos - 10), 
                                            end_time - start_time, 20,
                                            facecolor='orange', alpha=0.3, 
                                            edgecolor='orange', linewidth=1)
                    self.plot_ax.add_patch(rect)
                    
                    # Add annotation with missed details
                    annotation_text = f"Miss: W{period['wanted']} G{period['got']}"
                    self.plot_ax.annotate(annotation_text, 
                                        xy=(start_time + (end_time - start_time)/2, y_pos),
                                        xytext=(5, 5), textcoords='offset points',
                                        fontsize=6, color='darkorange', fontweight='bold',
                                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    def add_stimulus_markers(self):
        """Add vertical lines for stimulus events"""
        # Cue stimulus
        self.plot_ax.axvline(x=1000, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Cue Start')
        self.plot_ax.axvline(x=1200, color='green', linestyle=':', alpha=0.7, linewidth=2, label='Cue End')
        
        # Go signal
        self.plot_ax.axvline(x=3000, color='purple', linestyle='--', alpha=0.7, linewidth=2, label='Go Start')
        self.plot_ax.axvline(x=3100, color='purple', linestyle=':', alpha=0.7, linewidth=2, label='Go End')
        
        # Add stimulus legend
        stimulus_legend = [
            plt.Line2D([0], [0], color='green', linestyle='--', label='Cue'),
            plt.Line2D([0], [0], color='purple', linestyle='--', label='Go Signal')
        ]
        
        # Place stimulus legend in lower right
        self.plot_ax.legend(handles=stimulus_legend, loc='lower right', fontsize=8)


def load_latest_results():
    """Load the most recent analysis results"""
    # Look for recent analysis files
    analysis_files = list(Path('.').glob('multiple_ga_analysis_*.pkl'))
    if not analysis_files:
        print("❌ No analysis files found. Run the analysis notebook first.")
        return None, None
    
    # Get the most recent file
    latest_file = max(analysis_files, key=os.path.getctime)
    print(f"📁 Loading latest analysis results from: {latest_file}")
    
    with open(latest_file, 'rb') as f:
        data = pickle.load(f)
    
    target_dnas = data.get('target_dnas', [])
    simulation_results = data.get('simulation_results', [])
    
    if not target_dnas:
        print("❌ No target DNAs found in results file.")
        return None, None
    
    print(f"✅ Loaded {len(target_dnas)} target DNAs with simulation results")
    return target_dnas, simulation_results


def main():
    """Main function to launch the standalone DNA browser"""
    print("🚀 Starting Standalone DNA Browser...")
    print("This will open interactive plots in separate windows to avoid Jupyter widget issues.\n")
    
    # Load the latest results
    target_dnas, simulation_results = load_latest_results()
    
    if target_dnas is None:
        print("\n💡 To generate analysis results, run the analyze_multiple_ga_results.ipynb notebook first.")
        return
    
    # Create and launch the browser
    print(f"🎛️ Launching DNA Browser with {len(target_dnas)} target DNAs...")
    print("\nControls:")
    print("• DNA Slider: Browse through different DNA solutions")
    print("• Radio Buttons: Choose Experimental, Control, or Both conditions")
    print("• Network Button: Open network topology in new window")
    print("• Voltage plots show missed scoring with orange highlights")
    print("• 🎯 indicates DNAs meeting target criteria\n")
    
    browser = StandaloneDNABrowser(target_dnas, simulation_results, 
                                  pruning_threshold=975, max_weights=18)
    
    print("✅ DNA Browser launched! Close the plot window to exit.")


if __name__ == "__main__":
    main()