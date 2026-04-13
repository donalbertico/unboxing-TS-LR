import os
import numpy as np
import mne
import re
from mne import read_epochs, concatenate_epochs
from joblib import Parallel, delayed
from pyriemann.estimation import Covariances
from sklearn.preprocessing import LabelEncoder
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
import pickle

# Configuration
CLASSES = ["stop", "thank you"]
FILTER_SETS = [
    [(4, 8), (8, 13), (13, 30)],  # Set 1
    [(2, 40)],                   # Set 2
    [(8, 30)],                   # Set 3
    [(50, 120)],                 # Set 4
    [(2, 120)]                   # Set 5
]

def run_cv_final_filters(epochs_data, labels, groups):
    cov = Covariances(estimator='lwf')
    clf = LogisticRegression(max_iter=600, penalty='l1', solver='saga', n_jobs=1)
    cv = GroupKFold(n_splits=10)
    
    results = {i+1: [] for i in range(len(FILTER_SETS))}

    for i, f_set in enumerate(FILTER_SETS):
        # Pre-filter all bands for this set
        x_bands = []
        for low, high in f_set:
            data = epochs_data.copy().filter(low, high, verbose=False).crop(0.1, 1.8).get_data()
            x_bands.append(cov.fit_transform(data))
        
        # Cross-validation
        for train_idx, test_idx in cv.split(x_bands[0], labels, groups=groups):
            y_train, y_test = labels[train_idx], labels[test_idx]
            
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

    # Pool results
    final_accs = {}
    for f_idx, folds in results.items():
        if not folds: continue
        scs, szs = zip(*folds)
        final_accs[f_idx] = np.sum(np.array(scs) * np.array(szs)) / np.sum(szs)
    
    return final_accs

def get_data(sub):
    le = LabelEncoder()
    e1 = read_epochs(os.path.join(sub, 'gemini_ica_epo.fif'), verbose=False, preload=True)
    e2 = read_epochs(os.path.join(sub, 'gemini_validation_ica.fif'), verbose=False, preload=True)
    epochs = concatenate_epochs([e1, e2], verbose=False)
    if 'bads' in epochs.info and epochs.info['bads']:
        epochs.drop_channels(epochs.info['bads'])
    epochs_data = epochs[CLASSES].copy().pick('eeg').apply_baseline((-.3, 0), verbose=False)
    labels = le.fit_transform(epochs_data.events[:, 2])
    return epochs_data, labels

def process_subject(sub):
    if not os.path.exists(os.path.join(sub, 'gemini_ica_epo.fif')): return None
    try:
        epochs_data, labels = get_data(sub)
        groups = np.arange(len(labels)) // 4
        accs = run_cv_final_filters(epochs_data, labels, groups)
        return sub, accs
    except Exception as e:
        print(f'  Error in {sub}: {e}')
        return None

def run_all():
    subject_dirs = sorted([d for d in os.listdir('.') if os.path.isdir(d) and re.match(r'^s\d+$', d)])
    parallel_results = Parallel(n_jobs=15)(delayed(process_subject)(sub) for sub in subject_dirs)
    results = {r[0]: r[1] for r in parallel_results if r is not None}
    
    with open('bcicomp_final_filters_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print('Saved: bcicomp_final_filters_results.pkl')

if __name__ == "__main__":
    run_all()
