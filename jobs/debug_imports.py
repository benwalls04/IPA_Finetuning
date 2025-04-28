import sys
import os
import site

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Working directory: {os.getcwd()}")

print("\nEnvironment variables:")
for var in ['PATH', 'PYTHONPATH', 'CONDA_PREFIX', 'CONDA_DEFAULT_ENV']:
    print(f"{var}: {os.environ.get(var, 'Not set')}")

print("\nSite packages:")
for path in site.getsitepackages():
    print(f"  {path}")
    if os.path.exists(path):
        datasets_path = os.path.join(path, 'datasets')
        if os.path.exists(datasets_path):
            print(f"  ✓ datasets found at {datasets_path}")
        else:
            print(f"  ✗ datasets not found in {path}")

print("\nPython path:")
for i, path in enumerate(sys.path):
    print(f"  {i}: {path}")

print("\nTrying to import datasets:")
try:
    import datasets
    print(f"  ✓ datasets imported successfully (version: {datasets.__version__})")
    print(f"  ✓ datasets location: {datasets.__file__}")
except ImportError as e:
    print(f"  ✗ Failed to import datasets: {e}")
    print("\nDetailed traceback:")
    import traceback
    traceback.print_exc()
