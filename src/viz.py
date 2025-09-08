import plotly.graph_objs as go
from plotly.subplots import make_subplots
from IPython.display import display
import matplotlib.pyplot as plt
from src.constants import *
from src.constants import CRITERIA  # Explicit import for criteria highlighting
from src.neuron import *
from src.network import *
from src.validation import diagnose_conditions
from src.genetic_algorithm import *
from src.utils import *
from src.genetic_algorithm import *
import pandas as pd
import numpy as np

from src.constants import *

# def plot_neurons(neurons, sq_wave, go_wave):
#     fig, axs = plt.subplots(len(neurons), 1, figsize=(6, 3 * len(neurons)))
#     for i, neu in enumerate(neurons):
#         axs[i].plot(range(TMAX), neu.hist_V, label="V")
#         axs[i].plot(range(TMAX), neu.hist_u, label="u")
#         axs[i].plot(range(TMAX), sq_wave, label="SqWave", alpha=0.8, color="red", linestyle="dotted")
#         axs[i].plot(range(TMAX), go_wave / 5, label="GoWave", alpha=0.8, color="red", linestyle="dotted")
#         axs[i].set_title(f"{neu.name} dynamics")
#         axs[i].set_xlabel("ms")
#         axs[i].set_ylabel("mV")
#         axs[i].grid(True)
#         axs[i].legend(loc='upper right')
#     plt.tight_layout()
#     plt.show()

def display_matrix(matrix, nodes):
    assert matrix.shape == (len(nodes), len(nodes)), "Weight Matrix must be the same rank as the neuron name vector"
    df = pd.DataFrame(matrix, columns=nodes, index=nodes)
    display(df) 
    
def plot_neurons_interactive(hist_Vs, hist_us, neuron_names, sq_wave, go_wave, show_u=False, title=None, 
                           spike_raster=None, condition="experimental", show_missed_scoring=False):
    # print(f'{hist_Vs=}')
    # print(f'{neuron_names=}')
    assert len(hist_Vs) == len(neuron_names), "Must have the same number of neurons as the number of hist_Vs"
    n_neurons = len(neuron_names)
    n_cols = 1  # Set to 1 for a single column layout
    n_rows = n_neurons  # Each neuron gets its own row

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=neuron_names)

    hover_template = 'Time: %{x} ms<br>Value: %{y} mV'
    v_color = 'blue'  # Define a consistent color for hist_V
    u_color = 'orange'
    
    # Get missed scoring information if spike_raster is provided
    missed_periods = {}
    total_missed = 0
    if show_missed_scoring and spike_raster is not None:
        try:
            missed_diagnoses = diagnose_conditions(spike_raster, condition, return_list=True)
            # Group by neuron for easier lookup
            for diag in missed_diagnoses:
                neuron = diag['neuron']
                if neuron not in missed_periods:
                    missed_periods[neuron] = []
                missed_periods[neuron].append(diag)
            total_missed = len(missed_diagnoses)
        except Exception as e:
            print(f"Warning: Could not generate missed scoring diagnostics: {e}")
            missed_periods = {}

    for i in range(len(hist_Vs)):
        row = i + 1  # Adjust row index for single column
        col = 1
        neuron_name = neuron_names[i]
        
        # Check if this neuron is in CRITERIA_NAMES to determine if we should highlight it
        is_criteria_neuron = neuron_name in CRITERIA_NAMES
        
        # Add gold border for criteria neurons
        subplot_title = neuron_names[i]
        if is_criteria_neuron:
            subplot_title = f"🎯 {neuron_names[i]}"
        
        fig.add_trace(go.Scatter(x=list(range(TMAX)), y=hist_Vs[i], mode='lines', name='V',
                                 line=dict(color=v_color),  # Use the consistent color
                                 hovertemplate=hover_template), row=row, col=col)
        fig.add_trace(go.Scatter(x=list(range(TMAX)), y=hist_us[i], mode='lines', name='u',
                                 line=dict(color=u_color),  # Use the consistent color
                                 opacity=0.5,  # Set opacity at trace level
                                 hovertemplate=hover_template), row=row, col=col)
        # if show_u:
        #     fig.add_trace(go.Scatter(x=list(range(TMAX)), y=neu.hist_u, mode='lines', name='u',
        #                              line=dict(dash='dot', color='red'), hovertemplate=hover_template), row=row, col=col)
        fig.add_trace(go.Scatter(x=list(range(TMAX)), y=sq_wave, mode='lines', name='SqWave',
                                 line=dict(dash='dot', color='red'), hovertemplate=hover_template), row=row, col=col)
        fig.add_trace(go.Scatter(x=list(range(TMAX)), y=go_wave / 5, mode='lines', name='GoWave',
                                 line=dict(dash='dot', color='red'), hovertemplate=hover_template), row=row, col=col)
        
        # Add missed scoring highlighting for criteria neurons
        if show_missed_scoring and neuron_name in missed_periods:
            # Get voltage range for this neuron to properly position the highlighting
            v_min, v_max = min(hist_Vs[i]), max(hist_Vs[i])
            v_range = v_max - v_min
            highlight_bottom = v_min - v_range * 0.1
            highlight_top = v_max + v_range * 0.1
            
            for diag in missed_periods[neuron_name]:
                # Add orange highlighting for missed periods
                fig.add_shape(
                    type="rect",
                    x0=diag['t_start'], x1=diag['t_end'],
                    y0=highlight_bottom, y1=highlight_top,
                    fillcolor="orange", opacity=0.5,
                    line=dict(width=0),
                    row=row, col=col
                )
                
                # Add annotation showing expected vs actual
                annotation_text = f"Miss: W{diag['wanted']} G{diag['spikes']}"
                fig.add_annotation(
                    x=(diag['t_start'] + diag['t_end']) / 2,
                    y=highlight_top - v_range * 0.05,
                    text=annotation_text,
                    showarrow=False,
                    font=dict(size=8, color="darkorange"),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="orange",
                    borderwidth=1,
                    row=row, col=col
                )
        
        # Add criteria interval highlighting for criteria neurons
        if neuron_name in CRITERIA_NAMES:
            try:
                criteria_info = CRITERIA.get(condition, {}).get(neuron_name, {})
                
                if criteria_info and 'interval' in criteria_info:
                    interval = criteria_info['interval']
                    io_state = criteria_info.get('io', 'on')
                    
                    # Get voltage range for this neuron to properly position the highlighting
                    v_min, v_max = min(hist_Vs[i]), max(hist_Vs[i])
                    v_range = v_max - v_min
                    background_bottom = v_min - v_range * 0.05
                    background_top = v_max + v_range * 0.05
                    
                    # Add subtle orange background for criteria interval
                    fig.add_shape(
                        type="rect",
                        x0=interval[0], x1=interval[1],
                        y0=background_bottom, y1=background_top,
                        fillcolor="green", opacity=0.3,  # Light orange background
                        line=dict(width=0),
                        layer="below",  # Put behind other elements
                        row=row, col=col
                    )
                    
                    # Add a small annotation to indicate the criteria state
                    annotation_y = background_top - v_range * 0.02
                    state_text = f"Criteria: {io_state.upper()}"
                    
                    fig.add_annotation(
                        x=(interval[0] + interval[1]) / 2,
                        y=annotation_y,
                        text=state_text,
                        showarrow=False,
                        font=dict(size=8, color="darkorange"),
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor="orange",
                        borderwidth=1,
                        row=row, col=col
                    )
                    
            except Exception as e:
                # Silently skip if CRITERIA import fails
                pass

    # Update subplot titles to include missed scoring info and criteria neuron marking
    updated_titles = []
    for i, neuron_name in enumerate(neuron_names):
        is_criteria = neuron_name in CRITERIA_NAMES
        missed_count = len(missed_periods.get(neuron_name, []))
        
        if is_criteria and show_missed_scoring:
            title = f"🎯 {neuron_name} (Missed: {missed_count})"
        elif is_criteria:
            title = f"🎯 {neuron_name}"
        else:
            title = neuron_name
            
        updated_titles.append(title)
    
    # Update subplot titles
    for i, title in enumerate(updated_titles):
        fig.layout.annotations[i].text = title

    # Use the provided title if one is given, otherwise use the default
    title_text = title if title is not None else "Neuron Dynamics"
    if show_missed_scoring and total_missed > 0:
        title_text += f" (Total Missed: {total_missed} points)"
    
    fig.update_layout(height=300 * n_rows, width=900, title_text=title_text, showlegend=False)
    fig.show()
    return fig


# def plot_binned_differences(binned_differences):
#     tMax = int(len(binned_differences[0]) * BIN_SIZE)
#     time_intervals = np.arange(0, TMAX, BIN_SIZE)  # Create time intervals for the x-axis

#     n_neurons = int(len(binned_differences))
#     n_cols = 1  # Set to 1 for a single column layout
#     n_rows = n_neurons  # Each neuron gets its own row

#     fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[name for name in NEURON_NAMES])

#     hover_template = 'Time: %{x} ms<br>Value: %{y} spikes'
#     v_color = 'orange'  # Define a consistent color for spike times

#     # it would be nice to have 3 columns to show the experimental, control, and difference

#     for i, neu in enumerate(binned_differences):
#         row = i + 1  
#         col = 1
#         fig.add_trace(go.Bar(
#             x=time_intervals, 
#             y=neu, 
#             name=f'Neuron {i+1}',
#             marker=dict(color=v_color),  
#             hovertemplate=hover_template), 
#             row=row, 
#             col=col
#             )


#         # Calculate the maximum absolute value for symmetric y-axis for each neuron
#         max_abs_value = max(abs(neu.min()), abs(neu.max()))

#         # Update y-axis range for each subplot
#         fig.update_yaxes(range=[-max_abs_value, max_abs_value], row=row, col=col)

#     fig.update_layout(
#         height=300 * n_rows, 
#         width=900, 
#         title_text="Binned spikes: Experimental - Control", 
#         showlegend=False, 
#         bargap=0
#     )
#     fig.show()

def run_experiment(curr_dna, diag_list=[0,0,0,0]):
    dna_matrix = load_dna(curr_dna)

    # === Preparing Network === 
    all_neurons = create_neurons()
    splits, input_waves, alpha_array = create_experiment()
    criteria_dict = define_criteria()
    max_score = TMAX // BIN_SIZE * len(CRITERIA_NAMES)

    dna_score, neuron_data = evaluate_dna(
                    dna_matrix=dna_matrix,
                    neurons=all_neurons,
                    alpha_array=alpha_array,
                    input_waves=input_waves,
                    criteria=criteria_dict
                    )
    total_score = sum(dna_score.values())


    diagnostic = {
            'show_dna_matrix' : diag_list[0],
            'show_neuron_plots' : diag_list[1],
            'show_difference_histogram' : diag_list[2],
            'show_dna_scores': diag_list[3]
        }
    if diagnostic['show_dna_scores']:
                    print(f'    === DNA: {curr_dna}') 
                    print(f'    === Control: {dna_score["control"]}/{max_score}')
                    print(f'    === Experimental: {dna_score["experimental"]}/{max_score}')
                    print(f'    === Overall: {total_score}({total_score/(2*max_score):.2%})')
                    print('\n')

    if diagnostic['show_dna_matrix']:
                    print("Currently loaded matrix ---")
                    display_matrix(dna_matrix, NEURON_NAMES)

    if diagnostic['show_dna_scores']:
                    print(f'{dna_score=}: {curr_dna}')
                
    if diagnostic['show_neuron_plots']:
                    for condition in ['experimental', 'control']:
                        target_neurons_hist_Vs = np.array([neuron_data[condition][name]['hist_V'] for name in NEURON_NAMES])
                        target_neurons_hist_us = np.array([neuron_data[condition][name]['hist_u'] for name in NEURON_NAMES])
                        
                        plot_neurons_interactive(hist_Vs=target_neurons_hist_Vs, hist_us=target_neurons_hist_us, neuron_names=NEURON_NAMES, sq_wave=input_waves[0], go_wave=input_waves[1], show_u=False)
    return total_score
                    