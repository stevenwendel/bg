"""
Test script to verify DNA capture is working correctly
"""
import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.constants import GA_CONFIG
from ga_runner import run_ga

def test_dna_capture():
    """Test that DNA capture is working"""
    
    # Set up environment
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    
    # Create a small test configuration
    test_config = {
        "POP_SIZE": 10,
        "NUM_GENERATIONS": 3,
        "MUT_RATE": 0.3,
        "MUT_SIGMA": 0.5,
        "ELITE_SIZE": 2,
        "RANK_DEPTH": 5,
        "DNA_BOUNDS": [0, 500]
    }
    
    # Add to GA_CONFIG
    GA_CONFIG["test"] = test_config
    
    print("Testing DNA capture with small GA run...")
    print(f"Config: {test_config}")
    
    # Run GA and capture DNA
    best_score, best_dna = run_ga("test")
    
    print(f"\n✅ SUCCESS!")
    print(f"Best score: {best_score}")
    print(f"Best DNA length: {len(best_dna)}")
    print(f"Best DNA (first 5 values): {best_dna[:5]}")
    print(f"Best DNA (last 5 values): {best_dna[-5:]}")
    
    # Verify DNA is not None and has reasonable values
    assert best_dna is not None, "DNA should not be None"
    assert len(best_dna) > 0, "DNA should have length > 0"
    assert all(isinstance(x, (int, float)) for x in best_dna), "DNA should contain numbers"
    
    print(f"\n🎯 DNA capture is working correctly!")
    
    # Clean up
    del GA_CONFIG["test"]
    
    return best_score, best_dna

if __name__ == "__main__":
    test_dna_capture() 