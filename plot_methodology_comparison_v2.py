import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm

def load_pkl(path):
    if not os.path.exists(path):
        print(f"Warning: File {path} not found.")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

# Significance Threshold Functions
def p_adj(n):
    return (n * 0.5 + 2) / (n + 4)

def conf(n, p, alpha):
    return np.sqrt((p * (1 - p)) / (n + 4)) * norm.ppf(1 - (alpha / 2))

def getConf(n, alpha=0.01):
    p = p_adj(n)
    return p + conf(n, p, alpha)

def generate_methodology_v2():
    # 1. Load data
    bci_std = load_pkl('bci competition/Training set/bcicomp_complex_results_stop_thankyou_10_12.pkl')
    ng_std = load_pkl('nguyen/Short_Long_words/nguyen_complex_results_10_12.pkl')
    bci_new = load_pkl('bci competition/Training set/bcicomp_final_filters_results.pkl')
    ng_new = load_pkl('nguyen/Short_Long_words/nguyen_final_filters_results.pkl')

    # 2. Compute Significance Thresholds (Alpha = 0.01)
    # BCI: 137 relevant trials, Nguyen: 199 original trials
    thresh_bci = getConf(137, alpha=0.01)
    thresh_ng = getConf(199, alpha=0.01)
    print(f"Significance Thresholds (alpha=0.01): BCI={thresh_bci:.4f}, Nguyen={thresh_ng:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))

    # --- BCIComp Subplot ---
    if bci_std and bci_new:
        subjects_bci = sorted(list(bci_std.keys()))
        colors = plt.cm.tab20(np.linspace(0, 1, len(subjects_bci)))
        
        bci_group = [bci_std[s]['normal_1.8']['group'] for s in subjects_bci]
        bci_strat = [bci_std[s]['normal_1.8']['stratified'] for s in subjects_bci]
        bci_sanity = [bci_std[s]['sanity'] for s in subjects_bci]
        bci_fs1 = [bci_new[s][1] for s in subjects_bci if 1 in bci_new[s]]
        
        data_bci = [bci_group, bci_strat, bci_fs1, bci_sanity]
        labels = ['Group CV', 'Stratified CV', '<40Hz', 'Sanity Check']
        
        ax1.boxplot(data_bci, tick_labels=labels, showfliers=False)
        
        # Add colored jittered datapoints
        for sub_idx, sub in enumerate(subjects_bci):
            y_vals = [bci_std[sub]['normal_1.8']['group'], bci_std[sub]['normal_1.8']['stratified'], bci_new[sub][1], bci_std[sub]['sanity']]
            for i, y in enumerate(y_vals):
                x = np.random.normal(i + 1, 0.04)
                ax1.plot(x, y, color=colors[sub_idx], marker='o', markersize=4, alpha=0.6, linestyle='None')

        ax1.set_title("BCI Competition")
        ax1.set_ylabel("Pooled Accuracy")
        ax1.axhline(thresh_bci, color='red', linestyle='--', alpha=0.6, label=f'Significance (p<0.01)')
        ax1.set_ylim(0.4, 1.0)
        ax1.legend()
        ax1.tick_params(axis='x', labelsize=8)

    # --- Nguyen Subplot ---
    if ng_std and ng_new:
        subjects_ng = sorted(list(ng_std.keys()))
        colors_ng = plt.cm.tab10(np.linspace(0, 1, len(subjects_ng)))
        
        ng_group = [ng_std[s]['overlapping']['group_cv'] for s in subjects_ng]
        ng_strat = [ng_std[s]['overlapping']['stratified_cv'] for s in subjects_ng]
        ng_sanity = [ng_std[s]['sanity'] for s in subjects_ng]
        ng_fs1 = [ng_new[s][1] for s in subjects_ng if 1 in ng_new[s]]

        data_ng = [ng_group, ng_strat, ng_fs1, ng_sanity]

        ax2.boxplot(data_ng, tick_labels=labels, showfliers=False)

        # Add colored jittered datapoints
        for sub_idx, sub in enumerate(subjects_ng):
            y_vals = [ng_std[sub]['overlapping']['group_cv'], ng_std[sub]['overlapping']['stratified_cv'], ng_new[sub][1], ng_std[sub]['sanity']]
            for i, y in enumerate(y_vals):
                x = np.random.normal(i + 1, 0.04)
                ax2.plot(x, y, color=colors_ng[sub_idx], marker='o', markersize=4, alpha=0.6, linestyle='None')

        ax2.set_title("Nguyen")
        ax2.set_ylabel("")
        ax2.axhline(thresh_ng, color='red', linestyle='--', alpha=0.6, label=f'Significance (p<0.01)')
        ax2.set_ylim(0.4, 1.0)
        ax2.tick_params(axis='x', labelsize=8)

    os.makedirs("figures", exist_ok=True)
    save_path = 'figures/methodology_comparison_v2.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')

if __name__ == "__main__":
    generate_methodology_v2()
