import os
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

# Configuration
FILTER_SETS = [
    [(4, 8), (8, 13), (13, 30)],  # Set 1
    [(2, 40)],                   # Set 2
    [(8, 30)],                   # Set 3
    [(50, 80)],                  # Set 4 (max for Nguyen is 80)
    [(2, 80)]                    # Set 5
]

def get_overlapping_data(epochs_data, labels, f_set):
    """Returns covariances for overlapping splits for a given filter set."""
    windows = [(0, 2), (1.25, 3.25), (2.5, 4.5)]
    cov = Covariances(estimator='lwf')
    
    # Structure: [band_idx][trial_idx, channel, channel]
    # But wait, for overlapping splits, we concatenate the trials.
    # Original Nguyen logic: X_final = np.concatenate(all_bands_data, axis=1) 
    # where axis 1 is trials. 
    
    # We need to process each band in the filter set.
    x_bands_covs = []
    for low, high in f_set:
        band_data = epochs_data.copy().filter(low, high, verbose=False)
        segment_covs = []
        for tmin, tmax in windows:
            seg = band_data.copy().crop(tmin, tmax).get_data()
            segment_covs.append(cov.fit_transform(seg))
        # Concatenate across trial dimension (axis 0)
        x_bands_covs.append(np.concatenate(segment_covs, axis=0))
        
    y_final = np.tile(labels, len(windows))
    groups_final = np.tile(np.arange(len(labels)) // 3, len(windows))
    
    return x_bands_covs, y_final, groups_final

def run_cv_final_filters(epochs_data, labels):
    clf = LogisticRegression(max_iter=600, penalty='l1', solver='saga', n_jobs=1)
    cv = GroupKFold(n_splits=10)
    
    results = {i+1: [] for i in range(len(FILTER_SETS))}

    for i, f_set in enumerate(FILTER_SETS):
        x_bands, y, groups = get_overlapping_data(epochs_data, labels, f_set)
        
        for train_idx, test_idx in cv.split(x_bands[0], y, groups=groups):
            y_train, y_test = y[train_idx], y[test_idx]
            
            train_ts, test_ts = [], []
            for band_cov in x_bands:
                ts = TangentSpace()
                train_ts.append(ts.fit_transform(band_cov[train_idx]))
                test_ts.append(ts.transform(band_cov[test_idx]))
            
            X_tr = np.transpose(train_ts, (1, 0, 2)).reshape(len(y_train), -1)
            X_te = np.transpose(test_ts, (1, 0, 2)).reshape(len(y_test), -1)
            
            clf.fit(X_tr, y_train)
            score = clf.score(X_te, y_test)
            results[i+1].append((score, len(test_idx)))

    final_accs = {}
    for f_idx, folds in results.items():
        if not folds: continue
        scs, szs = zip(*folds)
        final_accs[f_idx] = np.sum(np.array(scs) * np.array(szs)) / np.sum(szs)
    
    return final_accs

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
        accs = run_cv_final_filters(epochs_data, labels)
        return sub, accs
    except Exception as e:
        print(f'  Error in {sub}: {e}')
        return None

def run_all():
    subject_dirs = sorted([d for d in os.listdir('.') if os.path.isdir(d) and re.match(r'^(s|sub)_\d+$', d)])
    parallel_results = Parallel(n_jobs=6)(delayed(process_subject)(sub) for sub in subject_dirs)
    results = {r[0]: r[1] for r in parallel_results if r is not None}
    
    with open('nguyen_final_filters_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print('Saved: nguyen_final_filters_results.pkl')

if __name__ == "__main__":
    run_all()
