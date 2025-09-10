# QUICK DATA CHECK: Add this to your notebook to see what data you have

def check_notebook_data():
    """Check what data is available in the notebook globals"""
    
    # Check what variables exist
    available_vars = [var for var in globals() if not var.startswith('_')]
    print("🔍 Available variables in notebook:")
    for var in sorted(available_vars):
        var_type = type(globals()[var])
        print(f"  {var}: {var_type}")
    
    # Check specifically for data we need
    data_vars = ['target_dnas', 'simulation_results', 'pruned_results', 'high_scoring_dnas']
    
    print("\n📊 Data variable check:")
    for var_name in data_vars:
        if var_name in globals():
            var_data = globals()[var_name]
            print(f"  ✅ {var_name}: {type(var_data)}, length: {len(var_data) if hasattr(var_data, '__len__') else 'N/A'}")
            
            if hasattr(var_data, '__len__') and len(var_data) > 0:
                first_item = var_data[0]
                print(f"    First item type: {type(first_item)}")
                if isinstance(first_item, dict):
                    print(f"    First item keys: {list(first_item.keys())}")
        else:
            print(f"  ❌ {var_name}: Not found")
    
    return available_vars

# Also create a simple test function
def test_with_available_data():
    """Test plotting with whatever data is available"""
    
    print("🧪 Testing with available data...")
    
    # Try different variable names
    dna_data = None
    sim_data = None
    
    # Check for target_dnas
    if 'target_dnas' in globals() and globals()['target_dnas']:
        dna_data = globals()['target_dnas']
        print(f"✅ Using target_dnas: {len(dna_data)} items")
    elif 'pruned_results' in globals() and globals()['pruned_results']:
        dna_data = globals()['pruned_results']
        print(f"✅ Using pruned_results: {len(dna_data)} items")
    elif 'high_scoring_dnas' in globals() and globals()['high_scoring_dnas']:
        dna_data = globals()['high_scoring_dnas']
        print(f"✅ Using high_scoring_dnas: {len(dna_data)} items")
    
    # Check for simulation results
    if 'simulation_results' in globals() and globals()['simulation_results']:
        sim_data = globals()['simulation_results']
        print(f"✅ Using simulation_results: {len(sim_data)} items")
    
    if dna_data and sim_data:
        print("🎯 Found both DNA and simulation data!")
        
        # Try to use the external window browser
        print("Importing external window browser...")
        exec(open('external_window_browser.py').read())
        
        print("Testing external window plot...")
        plot_dna_external_window(dna_data, sim_data, 0)
        
    elif dna_data:
        print("⚠️ Found DNA data but no simulation results")
        print("You may need to run the simulation generation step")
    else:
        print("❌ No suitable data found")
        print("Make sure you've run the analysis steps in your notebook")

print("✅ Data check functions loaded!")
print("Run: check_notebook_data()")
print("Run: test_with_available_data()")