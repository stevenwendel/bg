#!/usr/bin/env python3
"""
Generate Static DNA Plots - Creates PNG files for each DNA
No interactive widgets needed - just generates image files you can view
"""

import os
import sys
import pickle
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Import project modules
sys.path.append('.')

def generate_voltage_plot_image(dna_info, sim_results, dna_index, total_dnas, output_dir="dna_plots"):
    """Generate a voltage plot and save as PNG file"""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create figure
    fig, axes = plt.subplots(figsize=(16, 10))
    
    # Import constants
    try:
        from src.constants import NEURON_NAMES, CRITERIA_NAMES, TMAX
    except ImportError:
        print("❌ Cannot import constants. Make sure you're in the project root directory.")
        return None
    
    if not sim_results:
        axes.text(0.5, 0.5, 'No simulation results available', 
                 ha='center', va='center', transform=axes.transAxes, fontsize=14)
        plt.savefig(f"{output_dir}/dna_{dna_index+1:03d}_no_data.png", dpi=100, bbox_inches='tight')
        plt.close()
        return f"{output_dir}/dna_{dna_index+1:03d}_no_data.png"
    
    # Plot both conditions if available
    colors = {'exp': 'blue', 'cont': 'red'}
    condition_names = {'exp': 'Experimental', 'cont': 'Control'}
    
    conditions_to_plot = []
    if 'exp' in sim_results:
        conditions_to_plot.append('exp')
    if 'cont' in sim_results:
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
                         label=f"{condition_names[condition]}" if neuron_idx == 0 else "")
                
                # Add neuron label
                label_text = f"{neuron_name} ({condition_names[condition]})"
                if neuron_name in CRITERIA_NAMES:
                    label_text = "🎯 " + label_text
                
                axes.text(-200, y_pos, label_text, va='center', ha='right', fontsize=8)
                
                # Add missed scoring highlights if available
                if ('missed_scoring' in condition_data and 
                    neuron_name in condition_data['missed_scoring']):
                    
                    missed_info = condition_data['missed_scoring'][neuron_name]
                    for period in missed_info['missed_periods']:
                        start_time = period['start_time']
                        end_time = period['end_time']
                        
                        # Add orange highlight rectangle
                        from matplotlib.patches import Rectangle
                        rect = Rectangle((start_time, y_pos - 10), 
                                       end_time - start_time, 20,
                                       facecolor='orange', alpha=0.3, 
                                       edgecolor='orange', linewidth=1)
                        axes.add_patch(rect)
                        
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
    axes.set_xlabel('Time (ms)', fontsize=12)
    axes.set_ylabel('Voltage + Offset (mV)', fontsize=12)
    axes.set_yticks([])
    
    # Add title with DNA info
    title = (f"DNA {dna_index + 1}/{total_dnas} - "
            f"Score: {dna_info['pruned_score']} | "
            f"Weights: {dna_info['pruned_nonzero']} | "
            f"Reduction: {dna_info['weights_removed']/dna_info['original_nonzero']*100:.1f}%")
    
    if dna_info['pruned_score'] >= 975 and dna_info['pruned_nonzero'] <= 18:
        title = "🎯 " + title
        
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Add legend
    if len(conditions_to_plot) > 1:
        axes.legend(loc='upper right', fontsize=10)
    
    # Save the plot
    filename = f"{output_dir}/dna_{dna_index+1:03d}_score{dna_info['pruned_score']}_weights{dna_info['pruned_nonzero']}.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    
    return filename


def load_and_generate_plots():
    """Load data and generate all plots"""
    
    # Try to load from a simple test dataset first
    test_files = [
        "test_debug_fixed/aggregated_results.pkl",
        "test_debug_J/aggregated_results.pkl"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"📁 Loading test data from: {test_file}")
            
            try:
                with open(test_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Extract high-scoring DNAs
                high_scoring = []
                for dna_data in data:
                    if dna_data['score'] >= 980:  # Use a reasonable threshold
                        high_scoring.append(dna_data)
                
                if len(high_scoring) > 10:
                    high_scoring = high_scoring[:10]  # Limit to first 10
                
                print(f"✅ Found {len(high_scoring)} high-scoring DNAs")
                
                # Generate plots for each DNA
                from dna_simulation import run_dna_with_voltage_tracking
                
                for i, dna_data in enumerate(high_scoring):
                    print(f"📊 Generating plot {i+1}/{len(high_scoring)}...")
                    
                    # Create DNA info structure
                    dna_info = {
                        'pruned_dna': dna_data['dna'],
                        'pruned_score': dna_data['score'],
                        'pruned_nonzero': np.count_nonzero(dna_data['dna']),
                        'original_nonzero': np.count_nonzero(dna_data['dna']),
                        'weights_removed': 0
                    }
                    
                    # Run simulation
                    try:
                        exp_results = run_dna_with_voltage_tracking(dna_data['dna'], 'experimental')
                        cont_results = run_dna_with_voltage_tracking(dna_data['dna'], 'control')
                        
                        sim_results = {
                            'exp': exp_results,
                            'cont': cont_results
                        }
                        
                        # Generate plot
                        filename = generate_voltage_plot_image(dna_info, sim_results, i, len(high_scoring))
                        print(f"  ✅ Saved: {filename}")
                        
                    except Exception as e:
                        print(f"  ❌ Error generating plot {i+1}: {str(e)}")
                        continue
                
                print(f"\n✅ Generated plots in dna_plots/ directory")
                print("View the PNG files to see your DNA voltage traces!")
                return
                
            except Exception as e:
                print(f"❌ Error loading {test_file}: {str(e)}")
                continue
    
    print("❌ No suitable data files found.")
    print("Available files:")
    for file in Path('.').rglob('*.pkl'):
        print(f"  {file}")


def main():
    """Main function"""
    print("📊 DNA Static Plot Generator")
    print("Generates PNG files for DNA voltage traces - no interactive widgets needed!\n")
    
    load_and_generate_plots()
    
    print("\n💡 To view the plots:")
    print("1. Open the dna_plots/ directory")
    print("2. View the PNG files with any image viewer")
    print("3. Files are named: dna_XXX_scoreYYY_weightsZZZ.png")


if __name__ == "__main__":
    main()