# DEBUG VERSION: Add this cell to debug the data structure issues

import matplotlib.pyplot as plt
from IPython.display import clear_output
import ipywidgets as widgets
from ipywidgets import IntSlider, Dropdown, VBox, HBox, Output
import numpy as np

def debug_data_structure(target_dnas, simulation_results):
    """Debug function to check data structure"""
    print("🔍 DEBUGGING DATA STRUCTURE:")
    print(f"target_dnas type: {type(target_dnas)}")
    print(f"simulation_results type: {type(simulation_results)}")
    print(f"target_dnas length: {len(target_dnas) if target_dnas else 0}")
    print(f"simulation_results length: {len(simulation_results) if simulation_results else 0}")
    
    if target_dnas and len(target_dnas) > 0:
        print(f"\nFirst target_dna keys: {list(target_dnas[0].keys()) if isinstance(target_dnas[0], dict) else 'Not a dict'}")
        print(f"First target_dna: {target_dnas[0]}")
    
    if simulation_results and len(simulation_results) > 0:
        print(f"\nFirst simulation_result keys: {list(simulation_results[0].keys()) if isinstance(simulation_results[0], dict) else 'Not a dict'}")
        if isinstance(simulation_results[0], dict):
            for key, value in simulation_results[0].items():
                print(f"  {key}: {type(value)} - {list(value.keys()) if isinstance(value, dict) else str(value)[:100]}")

def create_simple_debug_browser(target_dnas, simulation_results):
    """Simple browser that shows debug info"""
    
    # Debug the data first
    debug_data_structure(target_dnas, simulation_results)
    
    if not target_dnas or len(target_dnas) == 0:
        print("❌ No target_dnas available")
        return widgets.HTML("No target_dnas available")
    
    if not simulation_results or len(simulation_results) == 0:
        print("❌ No simulation_results available")
        return widgets.HTML("No simulation_results available")
    
    # Create output widget for the plot
    plot_output = Output()
    
    # Create controls
    dna_slider = IntSlider(
        value=0, min=0, max=len(target_dnas)-1,
        description='DNA:'
    )
    
    def update_plot(dna_idx):
        """Simple update function with lots of debugging"""
        with plot_output:
            clear_output(wait=True)
            
            print(f"🔍 Updating plot for DNA index: {dna_idx}")
            
            try:
                current_dna = target_dnas[dna_idx]
                current_sim = simulation_results[dna_idx]
                
                print(f"Current DNA type: {type(current_dna)}")
                print(f"Current sim type: {type(current_sim)}")
                
                if isinstance(current_dna, dict):
                    print(f"DNA keys: {list(current_dna.keys())}")
                
                if isinstance(current_sim, dict):
                    print(f"Sim keys: {list(current_sim.keys())}")
                    
                    # Try to find voltage data
                    for condition in ['exp', 'cont']:
                        if condition in current_sim:
                            cond_data = current_sim[condition]
                            print(f"{condition} data type: {type(cond_data)}")
                            if isinstance(cond_data, dict):
                                print(f"{condition} keys: {list(cond_data.keys())}")
                                if 'voltage' in cond_data:
                                    voltage_dict = cond_data['voltage']
                                    print(f"{condition} voltage keys: {list(voltage_dict.keys()) if isinstance(voltage_dict, dict) else 'Not a dict'}")
                
                # Create a simple plot
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Try to plot something
                if isinstance(current_sim, dict) and 'exp' in current_sim:
                    exp_data = current_sim['exp']
                    if isinstance(exp_data, dict) and 'voltage' in exp_data:
                        voltage_dict = exp_data['voltage']
                        time_array = exp_data.get('time', np.arange(0, 5000, 1))
                        
                        y_pos = 0
                        plotted_any = False
                        for neuron_name, voltage in voltage_dict.items():
                            if isinstance(voltage, (list, np.ndarray)) and len(voltage) > 0:
                                ax.plot(time_array[:len(voltage)], np.array(voltage) + y_pos, 
                                       label=neuron_name, alpha=0.7)
                                y_pos += 50
                                plotted_any = True
                        
                        if plotted_any:
                            ax.set_xlabel('Time (ms)')
                            ax.set_ylabel('Voltage + Offset (mV)')
                            ax.legend()
                            ax.set_title(f"DNA {dna_idx + 1} - Debug View")
                        else:
                            ax.text(0.5, 0.5, 'No voltage data to plot', 
                                   ha='center', va='center', transform=ax.transAxes)
                    else:
                        ax.text(0.5, 0.5, 'No voltage data in exp condition', 
                               ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, 'No exp condition in simulation results', 
                           ha='center', va='center', transform=ax.transAxes)
                
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                print(f"❌ Error updating plot: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def on_change(change):
        update_plot(dna_slider.value)
    
    dna_slider.observe(on_change, names='value')
    
    # Create the interface
    browser = VBox([dna_slider, plot_output])
    
    # Initial plot
    update_plot(0)
    
    return browser

print("✅ Debug browser function created!")
print("Usage:")
print("debug_browser = create_simple_debug_browser(target_dnas, simulation_results)")
print("display(debug_browser)")