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
bci_new = load_pkl('bci competition/Training set/bcicomp_final_filters_results.pkl')
ng_new = load_pkl('nguyen/Short_Long_words/nguyen_final_filters_results.pkl')

def perform_analysis(std_data, new_data, ds_name, key_main):
    print(f"\n--- {ds_name} Statistical Analysis ---")
    
    subs = sorted(list(std_data.keys()))
    
    # Correct key extraction
    group_v, strat_v = [], []
    for s in subs:
        d = std_data[s][key_main]
        if ds_name == "BCI Competition":
            group_v.append(d['group'])
            strat_v.append(d['stratified'])
        else:
            group_v.append(d['group_cv'])
            strat_v.append(d['stratified_cv'])
            
    group_v = np.array(group_v)
    strat_v = np.array(strat_v)
    fs1_v   = np.array([new_data[s][1] for s in subs])
    sanity_v = np.array([std_data[s]['sanity'] for s in subs])
    
    labels = ['Group CV', 'Strat CV', 'FS1 <40Hz', 'Sanity']
    vectors = [group_v, strat_v, fs1_v, sanity_v]
    
    # Medians
    print("\nMedians:")
    for label, vec in zip(labels, vectors):
        print(f"  {label:12}: {np.median(vec):.4f}")
        
    # Pairwise T-test matrix (p-values)
    print("\nPairwise Paired T-test (p-values):")
    print(" " * 14 + "".join([f"{l:>12}" for l in labels]))
    for i, l1 in enumerate(labels):
        row_str = f"{l1:12} |"
        for j, l2 in enumerate(labels):
            if i == j:
                row_str += f"{'-':>12}"
            else:
                _, p = ttest_rel(vectors[i], vectors[j])
                row_str += f"{p:12.4f}"
        print(row_str)

if bci_std and bci_new:
    perform_analysis(bci_std, bci_new, "BCI Competition", 'normal_1.8')

if ng_std and ng_new:
    perform_analysis(ng_std, ng_new, "Nguyen Dataset", 'overlapping')
