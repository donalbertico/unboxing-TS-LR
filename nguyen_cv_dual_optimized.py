import os
import argparse
import numpy as np
import mne
import re
from mne import read_epochs
from joblib import Parallel, delayed
from pyriemann.estimation import Covariances
from sklearn.preprocessing import LabelEncoder
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
import pickle

# Setup
FREQ_STEP = 10
FREQ_SIZE = 12
MAX_FREQ_DEFAULT = 80

# Hyperparameters to test
CH_LIST = [2, 4, 8, 16, 24, 32, 64]
BAND_LIST = [1, 2, 4, 6, 8, 'all']

def get_possible_freqs(max_f, min_freq=2):
    frequency_ranges = []
    start_freq = min_freq
    while start_freq < max_f:
        end_freq = start_freq + FREQ_SIZE
        if end_freq <= max_f:
            frequency_ranges.append((start_freq, end_freq))
        start_freq += FREQ_STEP
    return sorted(list(set(frequency_ranges)))

def filter_data_to_numpy(epochs, tmin, tmax, max_f):
    x_all = []
    freqs = get_possible_freqs(max_f)
    for start, end in freqs:
        data = epochs.copy().filter(start, end, verbose=False).crop(tmin, tmax).get_data()
        x_all.append(data)
    return np.array(x_all)

def get_overlapping_data(epochs_data, labels, max_f):
    windows = [(0, 2), (1.25, 3.25), (2.5, 4.5)]
    all_bands_data = []
    for tmin, tmax in windows:
        all_bands_data.append(filter_data_to_numpy(epochs_data, tmin, tmax, max_f))
    
    x_final = np.concatenate(all_bands_data, axis=1)
    y_final = np.tile(labels, len(windows))
    groups_ids = np.arange(len(labels)) // 3
    groups_final = np.tile(groups_ids, len(windows))
    return x_final, y_final, groups_final

def map_ts_weights_to_channels_correctly(ts_weights, n_bands, n_channels):
    n_ts_features = len(ts_weights) // n_bands
    weights_reshaped = ts_weights.reshape(n_bands, n_ts_features)
    channel_weights = np.zeros((n_channels, n_bands))
    triu_idx = np.triu_indices(n_channels)
    for b in range(n_bands):
        band_w = weights_reshaped[b]
        for idx, (i, j) in enumerate(zip(*triu_idx)):
            val = np.abs(band_w[idx])
            if i != j: val *= np.sqrt(2)
            channel_weights[i, b] += val
            if i != j: channel_weights[j, b] += val
    return channel_weights

def run_cv_pooled_dual_optimized(x_filt_data, y, groups):
    unique_classes = np.unique(y)
    if len(unique_classes) < 2: return None

    cov = Covariances(estimator='lwf')
    clf = LogisticRegression(max_iter=600, penalty='l1', solver='saga', n_jobs=1)
    temp_lr = LogisticRegression(penalty='l1', solver='saga', max_iter=600, n_jobs=1)
    ts = TangentSpace()

    x_covs = [cov.fit_transform(band) for band in x_filt_data]
    n_bands = len(x_covs)
    n_orig_channels = x_covs[0].shape[1]

    cv = GroupKFold(n_splits=10)
    results_grid = {(ch, b): [] for ch in CH_LIST for b in BAND_LIST}

    for train_idx, test_idx in cv.split(x_covs[0], y, groups=groups):
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2: continue

        X_rank = [ts.fit_transform(band[train_idx]) for band in x_covs]
        X_rank_flat = np.transpose(X_rank, (1, 0, 2)).reshape(len(train_idx), -1)
        temp_lr.fit(X_rank_flat, y_train)

        weights = np.abs(temp_lr.coef_[0])
        ch_weights_matrix = map_ts_weights_to_channels_correctly(weights, n_bands, n_orig_channels)
        channel_importance = np.sum(ch_weights_matrix, axis=1)
        band_importance = np.sum(ch_weights_matrix, axis=0)
        
        sorted_ch_indices = np.argsort(channel_importance)[::-1]
        sorted_band_indices = np.argsort(band_importance)[::-1]

        for ch_count in CH_LIST:
            actual_chs = min(ch_count, n_orig_channels)
            top_ch_indices = sorted_ch_indices[:actual_chs]
            for band_count in BAND_LIST:
                actual_bands = n_bands if band_count == 'all' else min(band_count, n_bands)
                top_band_indices = sorted_band_indices[:actual_bands]
                
                x_reduced = [x_covs[b][:, top_ch_indices, :][:, :, top_ch_indices] for b in top_band_indices]
                
                train_ts, test_ts = [], []
                for band_data in x_reduced:
                    ts_inner = TangentSpace()
                    train_ts.append(ts_inner.fit_transform(band_data[train_idx]))
                    test_ts.append(ts_inner.transform(band_data[test_idx]))

                X_tr = np.transpose(train_ts, (1, 0, 2)).reshape(len(y_train), -1)
                X_te = np.transpose(test_ts, (1, 0, 2)).reshape(len(y_test), -1)

                clf.fit(X_tr, y_train)
                score = clf.score(X_te, y_test)
                results_grid[(ch_count, band_count)].append((score, len(test_idx)))

    final_pooled = {}
    for key, folds in results_grid.items():
        if not folds: 
            final_pooled[key] = np.nan
            continue
        scores, sizes = zip(*folds)
        final_pooled[key] = np.sum(np.array(scores) * np.array(sizes)) / np.sum(sizes)
    return final_pooled

def get_data(sub):
    le = LabelEncoder()
    epochs = read_epochs(os.path.join(sub, 'ica_epo.fif'), verbose=False, preload=True)
    if 'bads' in epochs.info and epochs.info['bads']:
        epochs.drop_channels(epochs.info['bads'])
    epochs_data = epochs.copy().pick('eeg')
    labels = le.fit_transform(epochs_data.events[:, 2])
    return epochs_data, labels

def process_subject(sub):
    if not os.path.exists(os.path.join(sub, 'ica_epo.fif')): return None
    try:
        epochs_data, labels = get_data(sub)
        x_final, y_final, g_final = get_overlapping_data(epochs_data, labels, MAX_FREQ_DEFAULT)
        acc_grid = run_cv_pooled_dual_optimized(x_final, y_final, g_final)
        return sub, acc_grid
    except Exception as e:
        print(f'  Error in {sub}: {e}')
        return None

def run_all():
    subject_dirs = sorted([d for d in os.listdir('.') if os.path.isdir(d) and re.match(r'^(s|sub)_\d+$', d)])
    parallel_results = Parallel(n_jobs=6)(delayed(process_subject)(sub) for sub in subject_dirs)
    results = {r[0]: r[1] for r in parallel_results if r is not None}
    
    with open('nguyen_dual_optimized_grid.pkl', 'wb') as f:
        pickle.dump(results, f)
    print('Saved: nguyen_dual_optimized_grid.pkl')

if __name__ == "__main__":
    run_all()
