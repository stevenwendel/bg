# SOLUTION: Add this cell to your Jupyter notebook to fix the slider update issue

import matplotlib.pyplot as plt
from IPython.display import clear_output
import ipywidgets as widgets
from ipywidgets import interact, IntSlider, Dropdown, VBox, HBox, Output, Button

def create_fixed_dna_browser(target_dnas, simulation_results, pruning_threshold=975, max_weights=18):
    """
    Fixed DNA browser that properly updates plots when slider changes
    The key fix: Use output widgets and explicit clearing
    """
    
    # Create output widget for the plot
    plot_output = Output()
    
    # Create controls
    dna_slider = IntSlider(
        value=0, min=0, max=len(target_dnas)-1,
        description='DNA:', style={'description_width': 'initial'}
    )
    
    condition_dropdown = Dropdown(
        options=[('Both', 'both'), ('Experimental', 'exp'), ('Control', 'cont')],
        value='both', description='Show:', style={'description_width': 'initial'}
    )
    
    def update_plot(dna_idx, condition):
        """Update the plot - this is the key fix"""
        with plot_output:
            # Clear previous output - THIS IS THE CRITICAL FIX
            clear_output(wait=True)
            
            # Get current data
            current_dna = target_dnas[dna_idx]
            current_sim = simulation_results[dna_idx]
            
            # Create new plot
            fig, ax = plt.subplots(figsize=(16, 8))
            
            # Determine which conditions to show
            show_exp = condition in ['both', 'exp']
            show_cont = condition in ['both', 'cont']
            
            # Plot using your existing visualization code
            if current_sim:
                y_offset = 0
                colors = {'exp': 'blue', 'cont': 'red'}
                condition_names = {'exp': 'Experimental', 'cont': 'Control'}
                
                conditions_to_plot = []
                if show_exp and 'exp' in current_sim:
                    conditions_to_plot.append('exp')
                if show_cont and 'cont' in current_sim:
                    conditions_to_plot.append('cont')
                
                for cond_idx, cond_key in enumerate(conditions_to_plot):
                    condition_data = current_sim[cond_key]
                    time_array = condition_data['time']
                    voltage_dict = condition_data['voltage']
                    
                    cond_offset = cond_idx * (len(NEURON_NAMES) + 1) * 20
                    
                    for neuron_idx, neuron_name in enumerate(NEURON_NAMES):
                        if neuron_name in voltage_dict:
                            voltage = voltage_dict[neuron_name]
                            y_pos = cond_offset + neuron_idx * 20
                            
                            # Plot voltage trace
                            ax.plot(time_array, voltage + y_pos, 
                                   color=colors[cond_key], alpha=0.7, linewidth=1,
                                   label=f"{condition_names[cond_key]}" if neuron_idx == 0 else "")
                            
                            # Add neuron label
                            label_text = f"{neuron_name} ({condition_names[cond_key]})"
                            if neuron_name in CRITERIA_NAMES:
                                label_text = "TARGET " + label_text
                            
                            ax.text(-200, y_pos, label_text, va='center', ha='right', fontsize=8)
                            
                            # Add missed scoring highlights
                            if ('missed_scoring' in condition_data and 
                                neuron_name in condition_data['missed_scoring']):
                                
                                missed_info = condition_data['missed_scoring'][neuron_name]
                                for period in missed_info['missed_periods']:
                                    start_time = period['start_time']
                                    end_time = period['end_time']
                                    
                                    # Orange highlight
                                    ax.axvspan(start_time, end_time, 
                                             ymin=(y_pos - 10) / (ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1000),
                                             ymax=(y_pos + 10) / (ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1000),
                                             alpha=0.3, color='orange')
                                    
                                    # Annotation
                                    annotation_text = f"Miss: W{period['wanted']} G{period['got']}"
                                    ax.annotate(annotation_text, 
                                              xy=(start_time + (end_time - start_time)/2, y_pos),
                                              xytext=(5, 5), textcoords='offset points',
                                              fontsize=6, color='darkorange', fontweight='bold',
                                              bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                
                # Add stimulus markers
                ax.axvline(x=1000, color='green', linestyle='--', alpha=0.7, linewidth=2)
                ax.axvline(x=1200, color='green', linestyle=':', alpha=0.7, linewidth=2)
                ax.axvline(x=3000, color='purple', linestyle='--', alpha=0.7, linewidth=2)
                ax.axvline(x=3100, color='purple', linestyle=':', alpha=0.7, linewidth=2)
                
                # Format plot
                ax.set_xlim(-300, TMAX + 300)
                ax.set_xlabel('Time (ms)', fontsize=10)
                ax.set_ylabel('Voltage + Offset (mV)', fontsize=10)
                ax.set_yticks([])
                
                # Title with DNA info
                title = (f"DNA {dna_idx + 1}/{len(target_dnas)} - "
                        f"Score: {current_dna['pruned_score']} | "
                        f"Weights: {current_dna['pruned_nonzero']} | "
                        f"Reduction: {current_dna['weights_removed']/current_dna['original_nonzero']*100:.1f}%")
                
                if current_dna['pruned_score'] >= pruning_threshold and current_dna['pruned_nonzero'] <= max_weights:
                    title = "TARGET " + title
                
                ax.set_title(title, fontsize=12, fontweight='bold')
                
                if len(conditions_to_plot) > 1:
                    ax.legend(loc='upper right')
            
            else:
                ax.text(0.5, 0.5, 'No simulation results available', 
                       ha='center', va='center', transform=ax.transAxes)
            
            plt.tight_layout()
            plt.show()
    
    # Connect the controls to the update function
    def on_change(change=None):
        update_plot(dna_slider.value, condition_dropdown.value)
    
    dna_slider.observe(on_change, names='value')
    condition_dropdown.observe(on_change, names='value')
    
    # Create the interface
    controls = HBox([dna_slider, condition_dropdown])
    browser = VBox([controls, plot_output])
    
    # Initial plot
    update_plot(0, 'both')
    
    return browser

# Usage: Replace your existing browser creation with:
# browser = create_fixed_dna_browser(target_dnas, simulation_results, PRUNING_THRESHOLD, MAX_PRUNED_WEIGHTS)
# display(browser)

print("✅ Fixed DNA browser function created!")
print("Replace your existing browser with:")
print("browser = create_fixed_dna_browser(target_dnas, simulation_results, PRUNING_THRESHOLD, MAX_PRUNED_WEIGHTS)")
print("display(browser)")