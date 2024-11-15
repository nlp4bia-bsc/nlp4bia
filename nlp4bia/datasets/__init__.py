
import os
import importlib

# Dynamically import all modules in the current folder
current_dir = os.path.dirname(__file__)
module_files = [f[:-3] for f in os.listdir(current_dir) if f.endswith(".py") and f != "__init__.py"]

# Import all modules
for module in module_files:
    importlib.import_module(f".{module}", package=__name__)

# Dynamically import sub-packages (e.g., utils, benchmark)
subpackages = [d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d)) and not d.startswith("__")]

for subpackage in subpackages:
    importlib.import_module(f".{subpackage}", package=__name__)
