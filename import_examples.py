"""
Examples of how to import from the src module in different contexts.

This file demonstrates various import strategies for accessing modules in the src/ directory.
"""

# ============================================================================
# METHOD 1: Add project root to Python path (Recommended for notebooks)
# ============================================================================
import sys
import os

# Add the project root directory to Python's module search path
project_root = os.path.dirname(os.path.abspath(__file__))  # Gets the directory containing this file
sys.path.append(project_root)

# Now you can import from src using absolute imports
from src.constants import TMAX, NEURON_NAMES, EPOCHS
from src.workbench import create_neurons, prepare_neurons
from src.neuron import Izhikevich, vectorised_step

print("Method 1 - Absolute imports with sys.path:")
print(f"TMAX: {TMAX}")
print(f"Neuron names: {NEURON_NAMES[:3]}...")  # Show first 3 names
print()

# ============================================================================
# METHOD 2: Direct import with explicit path (Alternative)
# ============================================================================
import importlib.util

# Load a module directly from its file path
spec = importlib.util.spec_from_file_location("constants", "src/constants.py")
constants_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(constants_module)

print("Method 2 - Direct module loading:")
print(f"TMAX: {constants_module.TMAX}")
print()

# ============================================================================
# METHOD 3: Using relative imports (Only works in proper package context)
# ============================================================================
"""
# This would only work if this file was part of a proper package structure
# and was run as a module, not as a script

# from .src.constants import TMAX  # Relative import (doesn't work in scripts)
# from ..src.constants import TMAX  # Parent relative import
"""

# ============================================================================
# METHOD 4: Environment variable approach
# ============================================================================
"""
# You can also set PYTHONPATH environment variable:
# export PYTHONPATH="/path/to/your/project:$PYTHONPATH"
# Then imports work normally:
# from src.constants import TMAX
"""

# ============================================================================
# PRACTICAL EXAMPLE: Using the imported modules
# ============================================================================
print("Practical example - Creating neurons:")
try:
    # Create neurons using the imported functions
    neurons = create_neurons()
    print(f"Created {len(neurons)} neurons")
    
    # Show some neuron properties
    for i, neuron in enumerate(neurons[:3]):  # Show first 3 neurons
        print(f"  {neuron.name}: idx={neuron.idx}")
        
except Exception as e:
    print(f"Error creating neurons: {e}")

print("\n" + "="*60)
print("IMPORT EXPLANATION:")
print("="*60)
print("""
Python Import System Fundamentals:
--------------------------------

1. MODULE SEARCH PATH:
   - Python looks for modules in directories listed in sys.path
   - Current directory (.) is usually first in the list
   - PYTHONPATH environment variable adds more directories

2. ABSOLUTE vs RELATIVE IMPORTS:
   - Absolute: from src.constants import TMAX
   - Relative: from .constants import TMAX (current package)
   - Relative: from ..constants import TMAX (parent package)

3. WHY RELATIVE IMPORTS FAIL IN NOTEBOOKS:
   - Jupyter notebooks run as __main__, not as part of a package
   - Relative imports require the module to be part of a package
   - The .. syntax means "go up two package levels" but notebooks aren't packages

4. BEST PRACTICES:
   - Use absolute imports with sys.path.append() for notebooks
   - Use proper package structure for production code
   - Set PYTHONPATH for development environments
""") 