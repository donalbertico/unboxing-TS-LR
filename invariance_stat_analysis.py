import pickle
import numpy as np
import os
from scipy.stats import ttest_rel

def load_pkl(path):
    if not os.path.exists(path): return None
    with open(path, 'rb') as f: return pickle.load(f)

# Load data
bci_std = load_pkl('bci competition/Training set/bcicomp_complex_results_stop_thankyou_10_12.pkl')
ng_std = load_pkl('nguyen/Short_Long_words/nguyen_complex_results_10_12.pkl')
bci_filt = load_pkl('bci competition/Training set/bcicomp_final_filters_results.pkl')
ng_filt = load_pkl('nguyen/Short_Long_words/nguyen_final_filters_results.pkl')

def perform_invariance_analysis(std_data, filt_data, ds_name, key_main):
    print(f"\n--- {ds_name} Invariance Statistical Analysis ---")
    
    subs = sorted(list(std_data.keys()))
    
    # 1. Extraction
    normal_v = []
    for s in subs:
        d = std_data[s][key_main]
        if ds_name == "BCI Competition":
            normal_v.append(d['group'])
        else:
            normal_v.append(d['group_cv'])
    
    normal_v = np.array(normal_v)
    fs_vectors = [np.array([filt_data[s][i] for s in subs]) for i in range(1, 6)]
    
    labels = ['Normal', 'FS1', 'FS2', 'FS3', 'FS4', 'FS5']
    all_vectors = [normal_v] + fs_vectors
    
    # 2. Medians
    print("\nMedians:")
    for label, vec in zip(labels, all_vectors):
        print(f"  {label:8}: {np.median(vec):.4f}")
        
    # 3. Pairwise T-test matrix (p-values)
    print("\nPairwise Paired T-test (p-values):")
    header = " " * 10 + "".join([f"{l:>10}" for l in labels])
    print(header)
    for i, l1 in enumerate(labels):
        row_str = f"{l1:8} |"
        for j, l2 in enumerate(labels):
            if i == j:
                row_str += f"{'-':>10}"
            else:
                _, p = ttest_rel(all_vectors[i], all_vectors[j])
                row_str += f"{p:10.4f}"
        print(row_str)

if bci_std and bci_filt:
    perform_invariance_analysis(bci_std, bci_filt, "BCI Competition", 'normal_1.8')

if ng_std and ng_filt:
    perform_invariance_analysis(ng_std, ng_filt, "Nguyen Dataset", 'overlapping')
