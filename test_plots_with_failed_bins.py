#!/usr/bin/env python3

import pickle
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Import the necessary modules from the project
from src.constants import *
from src.neuron import *
from src.network import *
from src.validation import *
from src.genetic_algorithm import *

# Import our updated plotting functions
from run_pruned_dna_interactive_plots import run_dna_with_corrected_voltage_tracking, plot_voltage_traces_interactive_with_borders, plot_voltage_traces_matplotlib_with_borders

def load_test_dna():
    """Load a high-scoring DNA from recent analysis for testing."""
    analysis_file = './multiple_ga_analysis_20250817_000644.pkl'
    
    with open(analysis_file, 'rb') as f:
        data = pickle.load(f)
    
    # Get the best DNA
    best_dna = data['high_scoring_dnas'][0]
    
    # Create a compatible format for our plotting functions
    test_individual = {
        'dna': best_dna['dna'],
        'total_score': best_dna['total_score'],
        'exp_score': best_dna['exp_score'],
        'cont_score': best_dna['cont_score'],
        'generation': best_dna['generation'],
        'final_nonzero': best_dna['non_zero_weights'],
        'original_nonzero': len(ACTIVE_SYNAPSES)  # Total possible connections
    }
    
    return test_individual

if __name__ == "__main__":
    print("Testing voltage trace plots with failed time bin indicators...")
    
    # Load test DNA
    test_individual = load_test_dna()
    
    print(f"Testing with DNA:")
    print(f"  Total fitness: {test_individual['total_score']}")
    print(f"  Experimental score: {test_individual['exp_score']}")
    print(f"  Control score: {test_individual['cont_score']}")
    print(f"  Non-zero weights: {test_individual['final_nonzero']}/{test_individual['original_nonzero']}")
    print(f"  Generation: {test_individual['generation']}")
    
    print(f"\nCRITERIA_NAMES (neurons used for fitness evaluation):")
    for i, name in enumerate(CRITERIA_NAMES):
        print(f"  {i+1:2d}. {name}")
    
    # Run the experiment with corrected voltage tracking
    print(f"\nRunning simulation...")
    results = run_dna_with_corrected_voltage_tracking(test_individual['dna'])
    
    print(f"\nSimulation completed!")
    print(f"Experimental spike count: {results['experimental']['spike_counts']}")
    print(f"Control spike count: {results['control']['spike_counts']}")
    
    # Show failed bins for debugging
    from src.validation import diagnose_conditions
    print(f"\nFailed bins for experimental condition:")
    diagnose_conditions(results['experimental']['spikes'], 'experimental', 20)
    
    print(f"\nFailed bins for control condition:")
    diagnose_conditions(results['control']['spikes'], 'control', 20)
    
    # Create interactive plotly plot with failed time bin indicators
    print(f"\nCreating interactive plot with failed time bin indicators...")
    fig_interactive = plot_voltage_traces_interactive_with_borders(results, test_individual)
    
    # Save interactive plot as HTML
    interactive_filename = './test_voltage_traces_with_failed_bins_interactive.html'
    fig_interactive.write_html(interactive_filename)
    print(f"Interactive plot saved to: {interactive_filename}")
    
    # Create static matplotlib plot with failed time bin indicators
    print(f"Creating static plot with failed time bin indicators...")
    fig_static = plot_voltage_traces_matplotlib_with_borders(results, test_individual)
    
    # Save static plot
    static_filename = './test_voltage_traces_with_failed_bins_static.png'
    plt.savefig(static_filename, dpi=300, bbox_inches='tight')
    print(f"Static plot saved to: {static_filename}")
    
    # Save the results
    results['test_dna_info'] = test_individual
    with open('./test_results_with_failed_bins.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Results saved to: test_results_with_failed_bins.pkl")
    
    print("\n🎉 VOLTAGE TRACE TEST WITH FAILED TIME BIN INDICATORS COMPLETE! 🎉")
    print("="*80)
    print("New visualization features:")
    print("  • 🟧 ORANGE HIGHLIGHTS show time bins that failed scoring criteria")
    print("  • Interactive plots show hover text: 'FAILED: wanted X, got Y'")
    print("  • Static plots show failed bins with orange backgrounds")
    print("  • All plots still include:")
    print("    - Fixed y-axis (-100 to 100 mV) for all plots")
    print("    - 🟨 Gold borders highlight neurons in CRITERIA_NAMES")
    print("    - 🟥 Red areas show stimulus periods (cue)")
    print("    - 🟩 Green areas show stimulus periods (go signal)")
    print("\nExample interpretation:")
    print("  • SNR1 should be 'off' (no spikes) during 1000-1200ms in experimental condition")
    print("  • If SNR1 shows spikes during that period, you'll see an orange highlight")
    print("  • This directly shows where the network is losing points in the fitness score!")
    print("="*80)