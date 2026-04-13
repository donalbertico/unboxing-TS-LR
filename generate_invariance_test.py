import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import norm

# Data Paths
BCI_DUAL = 'bci competition/Training set/bcicomp_dual_optimized_grid.pkl'
NG_DUAL = 'nguyen/Short_Long_words/nguyen_dual_optimized_grid.pkl'
BCI_STD = 'bci competition/Training set/bcicomp_complex_results_stop_thankyou_10_12.pkl'
NG_STD = 'nguyen/Short_Long_words/nguyen_complex_results_10_12.pkl'
BCI_FILTERS = 'bci competition/Training set/bcicomp_final_filters_results.pkl'
NG_FILTERS = 'nguyen/Short_Long_words/nguyen_final_filters_results.pkl'

# Params for Dual Grid
CH_LIST = [2, 4, 8, 16, 24, 32, 64]
BAND_LIST = [1, 2, 4, 6, 8, 'all']

# Significance Threshold Functions
def p_adj(n): return (n * 0.5 + 2) / (n + 4)
def conf(n, p, alpha): return np.sqrt((p * (1 - p)) / (n + 4)) * norm.ppf(1 - (alpha / 2))
def getConf(n, alpha=0.01):
    p = p_adj(n)
    return p + conf(n, p, alpha)

def load_pkl(path):
    if not os.path.exists(path): return None
    with open(path, 'rb') as f: return pickle.load(f)

def generate_invariance_test():
    # 1. Load All Data
    bci_dual_data = load_pkl(BCI_DUAL)
    ng_dual_data = load_pkl(NG_DUAL)
    bci_std_data = load_pkl(BCI_STD)
    ng_std_data = load_pkl(NG_STD)
    bci_filt_data = load_pkl(BCI_FILTERS)
    ng_filt_data = load_pkl(NG_FILTERS)

    # Master Figure
    fig = plt.figure(figsize=(14, 8))
    
    # Use subfigures with a stronger height ratio to give heatmaps more room
    subfigs = fig.subfigures(2, 1, height_ratios=[.9, 1])

    # --- ROW 1: BOXPLOTS (Reduced Size) ---
    # Strong margins to make the plots smaller within the frame
    subfigs[0].subplots_adjust(left=0.12, right=0.88, top=0.9, bottom=0.1, wspace=0.2)
    axes_box = subfigs[0].subplots(1, 2)
    
    thresh_bci = getConf(137, alpha=0.01)
    thresh_ng = getConf(199, alpha=0.01)
    labels = ['Filter bank', 'FS1 (T/A/B)', 'FS2 (2-40)', 'FS3 (8-30)', 'FS4 (>50)', 'FS5 (Full)']

    # BCI Boxplot
    if bci_std_data and bci_filt_data:
        subjects_bci = sorted(list(bci_std_data.keys()))
        colors = plt.cm.tab20(np.linspace(0, 1, len(subjects_bci)))
        
        bci_normal = [bci_std_data[s]['normal_1.8']['group'] for s in subjects_bci]
        bci_fs = [[bci_filt_data[s][i] for s in subjects_bci if i in bci_filt_data[s]] for i in range(1, 6)]
        data_bci = [bci_normal] + bci_fs
        axes_box[0].boxplot(data_bci, tick_labels=labels, widths=0.4, showfliers=False)
        
        # Add colored jittered datapoints
        for sub_idx, sub in enumerate(subjects_bci):
            y_vals = [bci_std_data[sub]['normal_1.8']['group']] + [bci_filt_data[sub][i] for i in range(1, 6)]
            for i, y in enumerate(y_vals):
                x = np.random.normal(i + 1, 0.04)
                axes_box[0].plot(x, y, color='black', marker='o', markersize=4, alpha=0.6, linestyle='None')

        axes_box[0].set_title("BCIComp: Invariance Test (Filter Sets)")
        axes_box[0].axhline(thresh_bci, color='red', linestyle='--', alpha=0.6, label=f'Signif (p<0.01): {thresh_bci:.3f}')
        axes_box[0].set_ylabel("Pooled Accuracy")
        axes_box[0].set_ylim(0.4, 1.0)
        axes_box[0].tick_params(axis='x', labelsize=8)

    # Nguyen Boxplot
    if ng_std_data and ng_filt_data:
        subjects_ng = sorted(list(ng_std_data.keys()))
        colors_ng = plt.cm.tab10(np.linspace(0, 1, len(subjects_ng)))
        
        ng_normal = [ng_std_data[s]['overlapping']['group_cv'] for s in subjects_ng]
        ng_fs = [[ng_filt_data[s][i] for s in subjects_ng if i in ng_filt_data[s]] for i in range(1, 6)]
        data_ng = [ng_normal] + ng_fs
        axes_box[1].boxplot(data_ng, tick_labels=labels, widths=0.4, showfliers=False)
        
        # Add colored jittered datapoints
        for sub_idx, sub in enumerate(subjects_ng):
            y_vals = [ng_std_data[sub]['overlapping']['group_cv']] + [ng_filt_data[sub][i] for i in range(1, 6)]
            for i, y in enumerate(y_vals):
                x = np.random.normal(i + 1, 0.04)
                axes_box[1].plot(x, y, color='black', marker='o', markersize=4, alpha=0.6, linestyle='None')

        axes_box[1].set_title("Nguyen: Invariance Test (Filter Sets)")
        axes_box[1].axhline(thresh_ng, color='red', linestyle='--', alpha=0.6, label=f'Signif (p<0.01): {thresh_ng:.3f}')
        axes_box[1].set_ylabel("Pooled Accuracy")
        axes_box[1].set_ylim(0.4, 1.0)
        axes_box[1].tick_params(axis='x', labelsize=8)
        axes_box[1].legend(loc='upper right')

    # --- ROW 2: DUAL SELECTOR HEATMAPS (Full Frame) ---
    # Minimal margins to maximize heatmap area
    subfigs[1].subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2, wspace=0.25)
    axes_dual = subfigs[1].subplots(1, 2)
    
    def plot_dual_grid(data, title, ax):
        subjects = list(data.keys())
        acc_matrix = np.zeros((len(BAND_LIST), len(CH_LIST)))
        for i, b in enumerate(BAND_LIST):
            for j, ch in enumerate(CH_LIST):
                all_sub_accs = [data[s][(ch, b)] for s in subjects if (ch, b) in data[s]]
                acc_matrix[i, j] = np.nanmean(all_sub_accs) if all_sub_accs else 0
        sns.heatmap(acc_matrix, annot=True, fmt=".3f", cmap='viridis', 
                    xticklabels=CH_LIST, yticklabels=BAND_LIST, ax=ax, square=False,
                    vmin=0.5, vmax=0.75,
                    cbar_kws={'label': 'Mean Pooled Accuracy'})
        ax.set_title(title)
        ax.set_xlabel("Selected Channels")
        ax.set_ylabel("Selected Bands")

    if bci_dual_data: plot_dual_grid(bci_dual_data, "BCIComp: Dual Selection Grid", axes_dual[0])
    if ng_dual_data: plot_dual_grid(ng_dual_data, "Nguyen: Dual Selection Grid", axes_dual[1])


    os.makedirs("figures", exist_ok=True)
    save_path = "figures/invariance_test.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Invariance test figure saved to: {save_path}")

if __name__ == "__main__":
    generate_invariance_test()
