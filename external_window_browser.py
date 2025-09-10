# EXTERNAL WINDOW VERSION: Creates plots in separate matplotlib windows
# Add this to your notebook and run it

import matplotlib
matplotlib.use('MacOSX')  # Force MacOSX backend for separate windows
import matplotlib.pyplot as plt
import numpy as np
from src.constants import NEURON_NAMES, CRITERIA_NAMES, TMAX

def plot_dna_external_window(target_dnas, simulation_results, dna_index=0):
    """
    Plot a specific DNA in an external matplotlib window
    This bypasses Jupyter widget issues completely
    """
    
    if not target_dnas or dna_index >= len(target_dnas):
        print(f"❌ Invalid DNA index {dna_index}. Available: 0-{len(target_dnas)-1 if target_dnas else 0}")
        return
    
    if not simulation_results or dna_index >= len(simulation_results):
        print(f"❌ Invalid simulation index {dna_index}. Available: 0-{len(simulation_results)-1 if simulation_results else 0}")
        return
    
    current_dna = target_dnas[dna_index]
    current_sim = simulation_results[dna_index]
    
    print(f"🖼️ Opening DNA {dna_index + 1} in external window...")
    
    # Create new figure - this will open in separate window
    plt.figure(figsize=(16, 10))
    ax = plt.gca()
    
    if not current_sim:
        ax.text(0.5, 0.5, 'No simulation results available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        plt.show()
        return
    
    # Plot conditions
    colors = {'exp': 'blue', 'cont': 'red'}
    condition_names = {'exp': 'Experimental', 'cont': 'Control'}
    
    conditions_to_plot = []
    if 'exp' in current_sim:
        conditions_to_plot.append('exp')
    if 'cont' in current_sim:
        conditions_to_plot.append('cont')
    
    if not conditions_to_plot:
        ax.text(0.5, 0.5, 'No exp or cont conditions found in simulation results', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        plt.show()
        return
    
    for cond_idx, cond_key in enumerate(conditions_to_plot):
        condition_data = current_sim[cond_key]
        
        if not isinstance(condition_data, dict):
            continue
            
        time_array = condition_data.get('time', np.arange(0, TMAX, 1))
        voltage_dict = condition_data.get('voltage', {})
        
        if not voltage_dict:
            continue
        
        cond_offset = cond_idx * (len(NEURON_NAMES) + 1) * 20
        
        for neuron_idx, neuron_name in enumerate(NEURON_NAMES):
            if neuron_name in voltage_dict:
                voltage = voltage_dict[neuron_name]
                
                if not isinstance(voltage, (list, np.ndarray)) or len(voltage) == 0:
                    continue
                
                y_pos = cond_offset + neuron_idx * 20
                
                # Ensure time and voltage arrays have same length
                min_len = min(len(time_array), len(voltage))
                time_plot = time_array[:min_len]
                voltage_plot = np.array(voltage[:min_len]) + y_pos
                
                # Plot voltage trace
                ax.plot(time_plot, voltage_plot, 
                       color=colors[cond_key], alpha=0.7, linewidth=1,
                       label=f"{condition_names[cond_key]}" if neuron_idx == 0 else "")
                
                # Add neuron label
                label_text = f"{neuron_name} ({condition_names[cond_key]})"
                if neuron_name in CRITERIA_NAMES:
                    label_text = "[TARGET] " + label_text
                
                ax.text(-200, y_pos, label_text, va='center', ha='right', fontsize=8)
                
                # Add missed scoring highlights if available
                if ('missed_scoring' in condition_data and 
                    neuron_name in condition_data['missed_scoring']):
                    
                    missed_info = condition_data['missed_scoring'][neuron_name]
                    for period in missed_info.get('missed_periods', []):
                        start_time = period.get('start_time', 0)
                        end_time = period.get('end_time', 0)
                        
                        if start_time < end_time:
                            # Orange highlight
                            ax.axvspan(start_time, end_time, 
                                     ymin=(y_pos - 10) / 500,
                                     ymax=(y_pos + 10) / 500,
                                     alpha=0.3, color='orange')
                            
                            # Annotation
                            wanted = period.get('wanted', '?')
                            got = period.get('got', '?')
                            annotation_text = f"Miss: W{wanted} G{got}"
                            ax.annotate(annotation_text, 
                                      xy=(start_time + (end_time - start_time)/2, y_pos),
                                      xytext=(5, 5), textcoords='offset points',
                                      fontsize=6, color='darkorange', fontweight='bold',
                                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Add stimulus markers
    ax.axvline(x=1000, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Cue Start')
    ax.axvline(x=1200, color='green', linestyle=':', alpha=0.7, linewidth=2, label='Cue End')
    ax.axvline(x=3000, color='purple', linestyle='--', alpha=0.7, linewidth=2, label='Go Start')
    ax.axvline(x=3100, color='purple', linestyle=':', alpha=0.7, linewidth=2, label='Go End')
    
    # Format plot
    ax.set_xlim(-300, TMAX + 300)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Voltage + Offset (mV)', fontsize=12)
    ax.set_yticks([])
    
    # Title with DNA info
    score = current_dna.get('pruned_score', current_dna.get('score', '?'))
    weights = current_dna.get('pruned_nonzero', current_dna.get('nonzero_weights', '?'))
    original_weights = current_dna.get('original_nonzero', weights)
    weights_removed = current_dna.get('weights_removed', 0)
    
    title = (f"DNA {dna_index + 1}/{len(target_dnas)} - "
            f"Score: {score} | Weights: {weights}")
    
    if isinstance(original_weights, (int, float)) and original_weights > 0:
        reduction_pct = (weights_removed / original_weights) * 100 if weights_removed > 0 else 0
        title += f" | Reduction: {reduction_pct:.1f}%"
    
    # Check if it's a target DNA
    is_target = (isinstance(score, (int, float)) and score >= 975 and 
                isinstance(weights, (int, float)) and weights <= 18)
    if is_target:
        title = "[TARGET] " + title
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    
    # Add legends
    if len(conditions_to_plot) > 1:
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    return plt.gcf()

def browse_dnas_external(target_dnas, simulation_results, start_index=0, count=5):
    """
    Open multiple DNA plots in separate windows
    """
    if not target_dnas:
        print("❌ No target_dnas available")
        return
    
    end_index = min(start_index + count, len(target_dnas))
    
    print(f"🖼️ Opening DNAs {start_index + 1} to {end_index} in separate windows...")
    
    for i in range(start_index, end_index):
        plot_dna_external_window(target_dnas, simulation_results, i)

print("✅ External window browser functions created!")
print("\nUsage:")
print("# Plot single DNA in external window:")
print("plot_dna_external_window(target_dnas, simulation_results, 0)")
print("")
print("# Browse multiple DNAs (opens separate windows):")
print("browse_dnas_external(target_dnas, simulation_results, 0, 3)  # Show first 3 DNAs")
print("")
print("# Quick access functions:")
print("def show_dna(n): plot_dna_external_window(target_dnas, simulation_results, n)")
print("def show_top_5(): browse_dnas_external(target_dnas, simulation_results, 0, 5)")