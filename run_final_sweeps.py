import os
import subprocess
from joblib import Parallel, delayed

bci_dir = os.path.join("bci competition", "Training set")
ng_dir = os.path.join("nguyen", "Short_Long_words")

def run_bci_final():
    print("--> Launching Final Filter Sweep (BCIComp, 15 cores)")
    cmd = "python run_final_filter_sweep_bci.py"
    subprocess.run(cmd, shell=True, cwd=bci_dir)

def run_ng_final():
    print("--> Launching Final Filter Sweep (Nguyen, 6 cores)")
    cmd = "python run_final_filter_sweep_nguyen.py"
    subprocess.run(cmd, shell=True, cwd=ng_dir)

if __name__ == "__main__":
    print("Starting final plain verification sweeps for both datasets...")
    Parallel(n_jobs=2)(delayed(func)() for func in [run_bci_final, run_ng_final])
    print("\nAll final sweeps completed.")
