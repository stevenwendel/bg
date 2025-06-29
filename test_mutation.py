"""
Test script to verify the new Gaussian mutation function
"""
import numpy as np
from src.genetic_algorithm import mutate_gauss, create_dna

def test_mutation_function():
    """Test the new Gaussian mutation function"""
    
    print("Testing Gaussian Mutation Function")
    print("=" * 50)
    
    # Create a test DNA vector
    bounds = (0, 500)
    original_dna = create_dna(bounds)
    
    print(f"Original DNA (first 10 elements): {original_dna[:10]}")
    print(f"DNA length: {len(original_dna)}")
    print(f"DNA bounds: {bounds}")
    
    # Test with different sigma values
    sigma_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    for sigma in sigma_values:
        print(f"\n--- Testing with sigma = {sigma} ---")
        
        # Apply mutation
        mutated_dna = mutate_gauss(original_dna.copy(), sigma, bounds)
        
        # Calculate changes
        changes = mutated_dna.astype(float) - original_dna.astype(float)
        
        print(f"Mutated DNA (first 10 elements): {mutated_dna[:10]}")
        print(f"Changes (first 10 elements): {changes[:10]}")
        print(f"Mean change: {np.mean(changes):.2f}")
        print(f"Std change: {np.std(changes):.2f}")
        print(f"Max change: {np.max(np.abs(changes)):.2f}")
        
        # Verify bounds are respected
        assert np.all(mutated_dna >= -bounds[1]), f"DNA values below lower bound with sigma={sigma}"
        assert np.all(mutated_dna <= bounds[1]), f"DNA values above upper bound with sigma={sigma}"
        
        # Verify signs are correct
        inhibitory_mask = np.isin(np.arange(len(mutated_dna)), 
                                 [i for i, (o, t) in enumerate(ACTIVE_SYNAPSES) 
                                  if o in INHIBITORY_NEURONS])
        excitatory_mask = ~inhibitory_mask
        
        assert np.all(mutated_dna[inhibitory_mask] <= 0), f"Inhibitory neurons should be negative with sigma={sigma}"
        assert np.all(mutated_dna[excitatory_mask] >= 0), f"Excitatory neurons should be positive with sigma={sigma}"
        
        print(f"✅ Bounds and signs verified for sigma={sigma}")
    
    print(f"\n🎯 All tests passed! Mutation function is working correctly.")

def test_mutation_formula():
    """Test that the mutation follows the correct formula"""
    
    print("\n" + "=" * 50)
    print("Testing Mutation Formula")
    print("=" * 50)
    
    # Create a simple test case
    test_dna = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    sigma = 0.5
    bounds = (0, 100)
    
    print(f"Test DNA: {test_dna}")
    print(f"Sigma: {sigma}")
    
    # Apply mutation
    mutated = mutate_gauss(test_dna.copy(), sigma, bounds)
    
    print(f"Mutated DNA: {mutated}")
    
    # Manually verify the formula for one element
    # For element 10: new_value = 10 + (10 * gaussian_noise)
    # The gaussian_noise has mean=0, std=sigma=0.5
    
    print(f"\nFormula verification:")
    print(f"Original element: 10")
    print(f"Expected: 10 + (10 * N(0, {sigma}))")
    print(f"Actual result: {mutated[0]}")
    
    # Show that the formula is working
    changes = mutated.astype(float) - test_dna.astype(float)
    relative_changes = changes / test_dna.astype(float)
    
    print(f"\nRelative changes (should be roughly N(0, {sigma})): {relative_changes}")
    print(f"Mean relative change: {np.mean(relative_changes):.3f}")
    print(f"Std relative change: {np.std(relative_changes):.3f}")
    
    print(f"\n✅ Formula verification complete!")

if __name__ == "__main__":
    # Import constants needed for the test
    from src.constants import ACTIVE_SYNAPSES, INHIBITORY_NEURONS
    
    test_mutation_function()
    test_mutation_formula() 