import os
import subprocess
from joblib import Parallel, delayed

bci_dir = os.path.join("bci competition", "Training set")
ng_dir = os.path.join("nguyen", "Short_Long_words")

def run_bci():
    print("--> Launching Optimized BCIComp Sweep (15 subjects parallel)")
    cmd = "python run_on_val_dual_optimized.py"
    subprocess.run(cmd, shell=True, cwd=bci_dir)

def run_ng():
    print("--> Launching Optimized Nguyen Sweep (6 subjects parallel)")
    cmd = "python nguyen_cv_dual_optimized.py"
    subprocess.run(cmd, shell=True, cwd=ng_dir)

if __name__ == "__main__":
    print("Starting optimized dual-selection experiments for both datasets...")
    # BCI uses 15 cores, Nguyen uses 6 cores. Total 21 cores.
    Parallel(n_jobs=2)(delayed(func)() for func in [run_bci, run_ng])
    print("\nAll optimized experiments completed.")
