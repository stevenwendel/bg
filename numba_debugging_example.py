"""
Numba Debugging Example
=======================

This file demonstrates common issues when using Numba and how to fix them.
The main problems in your original notebook were:

1. Missing variable imports
2. Function signature mismatches  
3. Undefined variables in function scope
4. Incorrect array indexing

Let's go through each issue step by step.
"""

import numpy as np
from numba import njit
from time import perf_counter

# ============================================================================
# PROBLEM 1: Missing Variable Imports
# ============================================================================

print("PROBLEM 1: Missing Variable Imports")
print("=" * 50)

# ❌ WRONG: Variables not defined in scope
def wrong_function():
    # This will fail because 'a' is not defined
    try:
        result = a + 1  # NameError: name 'a' is not defined
        return result
    except NameError as e:
        print(f"❌ Error: {e}")

# ✅ CORRECT: Import or define variables
def correct_function():
    # Define variables in scope
    a = np.array([1, 2, 3], dtype=np.float32)
    result = a + 1
    print(f"✅ Success: {result}")
    return result

wrong_function()
correct_function()

# ============================================================================
# PROBLEM 2: Function Signature Mismatches
# ============================================================================

print("\nPROBLEM 2: Function Signature Mismatches")
print("=" * 50)

@njit
def add_arrays(a, b):
    """Simple function to add two arrays."""
    return a + b

# ❌ WRONG: Calling with wrong number of arguments
def wrong_call():
    try:
        a = np.array([1, 2, 3])
        result = add_arrays(a)  # TypeError: missing 1 required positional argument
        return result
    except TypeError as e:
        print(f"❌ Error: {e}")

# ✅ CORRECT: Call with correct arguments
def correct_call():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    result = add_arrays(a, b)
    print(f"✅ Success: {result}")
    return result

wrong_call()
correct_call()

# ============================================================================
# PROBLEM 3: Array Indexing Issues
# ============================================================================

print("\nPROBLEM 3: Array Indexing Issues")
print("=" * 50)

@njit
def array_operation(arr):
    """Demonstrate correct array indexing."""
    n = arr.shape[0]
    result = np.zeros_like(arr)
    
    for i in range(n):
        result[i] = arr[i] * 2
    
    return result

# ❌ WRONG: Incorrect array indexing
def wrong_indexing():
    try:
        arr = np.zeros((3, 4))
        # This would cause issues in Numba
        result = arr[:, 0]  # Numba prefers explicit loops
        return result
    except Exception as e:
        print(f"❌ Potential issue with complex indexing: {e}")

# ✅ CORRECT: Explicit indexing
def correct_indexing():
    arr = np.array([1, 2, 3, 4, 5])
    result = array_operation(arr)
    print(f"✅ Success: {result}")
    return result

wrong_indexing()
correct_indexing()

# ============================================================================
# PROBLEM 4: Variable Scope in Numba Functions
# ============================================================================

print("\nPROBLEM 4: Variable Scope in Numba Functions")
print("=" * 50)

# ❌ WRONG: Using global variables in Numba function
global_var = 10

@njit
def wrong_global_usage():
    # This will fail because Numba can't access global variables
    try:
        return global_var + 1
    except Exception as e:
        print(f"❌ Numba can't access global variables: {e}")

# ✅ CORRECT: Pass variables as parameters
@njit
def correct_parameter_usage(value):
    return value + 1

def test_scope():
    try:
        wrong_global_usage()
    except:
        pass
    
    result = correct_parameter_usage(10)
    print(f"✅ Success with parameters: {result}")

test_scope()

# ============================================================================
# DEBUGGING TIPS
# ============================================================================

print("\nDEBUGGING TIPS")
print("=" * 50)

def debugging_tips():
    """Show how to debug Numba issues."""
    
    print("1. Start without @njit decorator:")
    print("   - Test your function with regular Python first")
    print("   - Make sure all variables are defined")
    print("   - Check that all imports are correct")
    
    print("\n2. Add @njit gradually:")
    print("   - Start with simple functions")
    print("   - Add complexity step by step")
    print("   - Use error messages to identify issues")
    
    print("\n3. Common Numba limitations:")
    print("   - No global variables")
    print("   - Limited Python features")
    print("   - Specific data types required")
    print("   - No dynamic typing")
    
    print("\n4. Use Numba's error messages:")
    print("   - They often point to the exact issue")
    print("   - Look for 'TypingError' messages")
    print("   - Check for 'NameError' in variable scope")

debugging_tips()

# ============================================================================
# PRACTICAL EXAMPLE: Fixed Version of Your Simulation
# ============================================================================

print("\nPRACTICAL EXAMPLE: Fixed Simulation")
print("=" * 50)

def create_simple_simulation():
    """Create a simplified version of your simulation to test."""
    
    # Define all parameters explicitly
    N = 5  # Number of neurons
    TMAX = 100  # Time steps
    
    # Neuron parameters (simplified)
    a = np.array([0.03] * N, dtype=np.float32)
    b = np.array([-2.0] * N, dtype=np.float32)
    C = np.array([100.0] * N, dtype=np.float32)
    
    # Weight matrix
    W = np.random.randn(N, N).astype(np.float32) * 0.1
    
    # Input waves
    cue_wave = np.zeros(TMAX, dtype=np.float32)
    cue_wave[10:20] = 1.0  # Simple cue
    
    @njit
    def simple_step(V, U, I, a, b, C, W):
        """Simplified neuron step."""
        n = V.size
        spk = np.zeros(n, dtype=np.uint8)
        
        for i in range(n):
            # Simple neuron dynamics
            dV = (I[i] - V[i]) / C[i]
            dU = a[i] * (b[i] - U[i])
            
            V[i] += dV
            U[i] += dU
            
            if V[i] > 0:
                V[i] = -60.0
                spk[i] = 1
        
        # Simple synaptic transmission
        if np.sum(spk) > 0:
            I += spk.astype(np.float32) @ W
        
        return spk
    
    @njit
    def simple_simulate(V, U, I, a, b, C, W, cue_wave, tmax):
        """Simplified simulation."""
        history = np.zeros((N, tmax), dtype=np.uint8)
        
        for t in range(tmax):
            I[0] += cue_wave[t]  # Add input to first neuron
            spk = simple_step(V, U, I, a, b, C, W)
            history[:, t] = spk
        
        return history
    
    # Run simulation
    V = np.full(N, -60.0, dtype=np.float32)
    U = np.zeros(N, dtype=np.float32)
    I = np.zeros(N, dtype=np.float32)
    
    history = simple_simulate(V, U, I, a, b, C, W, cue_wave, TMAX)
    
    print(f"✅ Simulation completed successfully!")
    print(f"   Shape: {history.shape}")
    print(f"   Total spikes: {np.sum(history)}")
    
    return history

# Run the example
try:
    history = create_simple_simulation()
    print("🎉 All examples completed successfully!")
except Exception as e:
    print(f"❌ Error in example: {e}")

print("\n" + "="*60)
print("SUMMARY: The main issues in your notebook were:")
print("1. Missing imports from src.constants")
print("2. Function signatures not matching between definition and calls")
print("3. Variables not defined in the notebook scope")
print("4. Incorrect array indexing in the step_kernel function")
print("="*60) 