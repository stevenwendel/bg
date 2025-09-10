# Add this cell to your Jupyter notebook to enable external plotting

# Import the helper functions
from simple_plot_viewer import create_notebook_helper_functions
plot_dna, browse_dnas, show_network = create_notebook_helper_functions()

print("✅ External plotting functions loaded!")
print("\nUsage:")
print("1. browse_dnas(target_dnas, simulation_results, DNA_INDEX)")
print("2. show_network(target_dnas, DNA_INDEX)")
print("3. plot_dna(dna_info, sim_results, index, total)")
print("\nExample:")
print("browse_dnas(target_dnas, simulation_results, 0)  # Show DNA 1")
print("show_network(target_dnas, 0)  # Show network for DNA 1")

# Example function to quickly browse through multiple DNAs
def show_top_dnas(n=5):
    """Show the first n DNAs in separate windows"""
    if 'target_dnas' in globals() and 'simulation_results' in globals():
        for i in range(min(n, len(target_dnas))):
            print(f"Showing DNA {i+1}...")
            browse_dnas(target_dnas, simulation_results, i)
    else:
        print("❌ target_dnas and simulation_results not found. Run the analysis first.")

print("\nQuick function: show_top_dnas(5) to show first 5 DNAs")