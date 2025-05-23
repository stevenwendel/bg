import os
import glob
import importlib

# Get all Python files in the current directory
modules = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
__all__ = [os.path.basename(f)[:-3] for f in modules if not f.endswith("__init__.py")]

# Import all modules
for module in __all__:
    importlib.import_module(f'src.{module}')