import pickle
import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths to the confirmed best config (sz=12, ch=64)
BCI_FILE = 'bci competition/Training set/bcicomp_chsel_stop_thankyou_sz12_ch64.pkl'
NG_FILE = 'nguyen/Short_Long_words/nguyen_chsel_sz12_ch64.pkl'

def get_possible_freqs(max_f, freq_step, freq_size, min_freq=2):
    frequency_ranges = []
    start_freq = min_freq
    while start_freq < max_f:
        end_freq = start_freq + freq_size
        if end_freq <= max_f:
            frequency_ranges.append((start_freq, end_freq))
        start_freq += freq_step
    unique_ranges = sorted(list(set(frequency_ranges)))
    band_names = [f"{r[0]}-{r[1]}Hz" for r in unique_ranges]
    return unique_ranges, band_names

def plot_topomap_grid(sub_subfigure, weights, info, band_names, title):
    n_bands = weights.shape[1]
    nrows = 3
    ncols = int(np.ceil(n_bands / nrows))
    
    # Use subplots
    axes = sub_subfigure.subplots(nrows, ncols)
    axes = axes.flatten()
    global_max = np.max(weights)
    im = None
    for b in range(n_bands):
        ax = axes[b]
        band_w = weights[:, b]
        im, _ = mne.viz.plot_topomap(
            band_w, info, axes=ax, cmap='Reds', vlim=(0, global_max),
            show=False, contours=4, extrapolate='local'
        )
        ax.set_title(band_names[b], fontsize=13)
    for empty_ax in axes[n_bands:]: empty_ax.axis('off')
    
    # Use fig.colorbar on the subfigure's im, pointing to the axes list
    cbar = sub_subfigure.colorbar(im, ax=axes.tolist(), shrink=0.8, location='right', aspect=30)
    cbar.set_label('Importance', fontsize=12)
    
    sub_subfigure.suptitle(title, fontsize=16)

def generate_compound_figure():
    # 1. Load Data
    with open(BCI_FILE, 'rb') as f: bci_data = pickle.load(f)
    with open(NG_FILE, 'rb') as f: ng_data = pickle.load(f)
    
    s05_weights = bci_data['s05']['weights']
    sub08_weights = ng_data['sub_08']['weights']


    # 2. Get Info for topomaps
    e_bci = mne.read_epochs("bci competition/Training set/s05/gemini_ica_epo.fif", verbose=False)
    if 'bads' in e_bci.info and e_bci.info['bads']: e_bci.drop_channels(e_bci.info['bads'])
    info_bci = e_bci.pick('eeg').info
    
    e_ng = mne.read_epochs("nguyen/Short_Long_words/sub_08/ica_epo.fif", verbose=False)
    if 'bads' in e_ng.info and e_ng.info['bads']: e_ng.drop_channels(e_ng.info['bads'])
    info_ng = e_ng.pick('eeg').info
    
    # Standardize weights to match info (handle potential alignment artifacts)
    if s05_weights.shape[0] > len(info_bci.ch_names): s05_weights = s05_weights[:len(info_bci.ch_names), :]
    if sub08_weights.shape[0] > len(info_ng.ch_names): sub08_weights = sub08_weights[:len(info_ng.ch_names), :]

    # 3. Compute Correlations
    corr_bci = np.corrcoef(s05_weights.T)
    corr_ng = np.corrcoef(sub08_weights.T)
    
    _, band_names_bci = get_possible_freqs(127, 10, 12)
    _, band_names_ng = get_possible_freqs(80, 10, 12)

    # 4. Create Master Figure with constrained layout
    fig = plt.figure(figsize=(16, 9))
    subfigs = fig.subfigures(2, 1, height_ratios=[1.2, .8])

    subfigs[0].subplots_adjust(left=-0.01, right=1.01, top=0.85, bottom=0.0, wspace=0)
    # subfigs[0].subplots_adjust(left=0, right=0, top=0.0, bottom=0., wspace=0.)
    # --- ROW 1: TOPOMAPS ---
    top_row_subfigs = subfigs[0].subfigures(1, 2)
    plot_topomap_grid(top_row_subfigs[0], s05_weights, info_bci, band_names_bci, "BCIComp p05: Spatial Importance")
    plot_topomap_grid(top_row_subfigs[1], sub08_weights, info_ng, band_names_ng, "Nguyen p08: Spatial Importance")

    # --- ROW 2: CORRELATIONS ---
    subfigs[1].subplots_adjust(left=0.13, right=0.92, top=0.85, bottom=0.25, wspace=0.4)
    axes_corr = subfigs[1].subplots(1, 2)
    
    # square=False ensures they fill the rectangular frame
    sns.heatmap(corr_bci, annot=True, fmt=".2f", cmap='viridis', 
                xticklabels=band_names_bci, yticklabels=band_names_bci, 
                ax=axes_corr[0], square=False, cbar_kws={'label': 'Pearson Correlation'})
    axes_corr[0].set_title("BCIComp p05: Band Correlation", fontsize=16)
    
    sns.heatmap(corr_ng, annot=True, fmt=".2f", cmap='viridis', 
                xticklabels=band_names_ng, yticklabels=band_names_ng, 
                ax=axes_corr[1], square=False, cbar_kws={'label': 'Pearson Correlation'})
    axes_corr[1].set_title("Nguyen p08: Band Correlation", fontsize=16)

    os.makedirs("figures", exist_ok=True)
    save_path = "figures/compound_analysis_s05_sub08.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Compound analysis figure saved to: {save_path}")

if __name__ == "__main__":
    generate_compound_figure()
