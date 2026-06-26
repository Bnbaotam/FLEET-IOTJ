#!/usr/bin/env python3

libraries = [
    "yacs",
    "tqdm",
    "numpy",
    "scipy",
    "torch",
    "optuna",
    "ot",        # Also imported as "ot", but leave as requested
    "flwr",
    "dill",
    "ncls",
    "orjson",
    "joblib",
    "matplotlib",
    "seaborn",
    "sklearn"     # scikit-learn imports as "sklearn"
]

print("\n=== Checking Python Library Imports ===\n")

for lib in libraries:
    try:
        __import__(lib)
        print(f"[OK]     {lib} imported successfully")
    except ImportError as e:
        print(f"[FAILED] {lib} failed to import → {e}")

print("\n=== Import Check Complete ===")
