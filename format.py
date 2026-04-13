import mne
import numpy as np
import pickle
import os
import re
from mne.preprocessing import ICA

def format_bcicomp():
    """
    Formats the BCI Competition dataset from raw_epo.fif to gemini_ica_epo.fif
    and gemini_validation_ica.fif using Picard ICA.
    """
    print("\n--- Formatting BCI Competition Dataset ---")
    
    # Load manual exclusions
    bad_data_path = 'bad_data_list.pkl'
    if not os.path.exists(bad_data_path):
        print("Warning: bad_data_list.pkl not found. Skipping manual exclusions.")
        exclude_dict = {}
    else:
        with open(bad_data_path, 'rb') as f:
            exclude_dict = pickle.load(f)

    # Process all sXX folders
    folders = sorted([d for d in os.listdir('.') if re.match(r'^s\d+$', d)])
    frontal_chs = ['Fp1', 'Fp2', 'AF7', 'AF8', 'AF3', 'AF4']

    for idx, folder in enumerate(folders):
        print(f"Processing {folder}...")
        
        # 1. Training Set
        train_raw_path = os.path.join(folder, 'raw_epo.fif')
        if os.path.exists(train_raw_path):
            train_epochs = mne.read_epochs(train_raw_path, preload=True, verbose=False)
            
            # Apply manual exclusions
            exclude_info = exclude_dict.get(idx, {'epochs': [], 'channels': []})
            train_epochs.info['bads'] = exclude_info.get('channels', [])
            valid_epochs = [e for e in exclude_info.get('epochs', []) if e < len(train_epochs)]
            train_epochs.drop(valid_epochs, verbose=False)

            # Fit ICA on pre-processed training copy
            ica = ICA(n_components=0.9, method='picard', random_state=42)
            train_copy = train_epochs.copy().pick('eeg')
            train_copy.apply_baseline((-.4, -.1), verbose=False)
            train_copy.crop(0, 1.5, verbose=False)
            train_copy.filter(1, 127, verbose=False)
            ica.fit(train_copy, verbose=False)

            # Identify and exclude blink component
            avail_frontal = [ch for ch in frontal_chs if ch in train_copy.ch_names]
            f_idx = [train_copy.ch_names.index(ch) for ch in avail_frontal]
            components = ica.get_components()
            f_scores = [np.mean(np.abs(components[f_idx, i])) for i in range(ica.n_components_)]
            blink_idx = int(np.argmax(f_scores))
            
            # Apply ICA to original training epochs
            ica.apply(train_epochs, exclude=[blink_idx], verbose=False)
            train_epochs.set_eeg_reference(ref_channels='average', verbose=False)
            
            out_train = os.path.join(folder, 'gemini_ica_epo.fif')
            train_epochs.save(out_train, overwrite=True, verbose=False)
            print(f"  Saved cleaned training epochs: {out_train}")

            # 2. Validation Set
            val_raw_path = os.path.join(folder, 'validation_raw_epo.fif')
            if os.path.exists(val_raw_path):
                val_epochs = mne.read_epochs(val_raw_path, preload=True, verbose=False)
                
                # Apply training bad channels and ICA component
                val_epochs.info['bads'] = exclude_info.get('channels', [])
                ica.apply(val_epochs, exclude=[blink_idx], verbose=False)
                val_epochs.set_eeg_reference(ref_channels='average', verbose=False)
                
                out_val = os.path.join(folder, 'gemini_validation_ica.fif')
                val_epochs.save(out_val, overwrite=True, verbose=False)
                print(f"  Saved cleaned validation epochs: {out_val}")

def format_nguyen():
    """
    The Nguyen dataset is typically provided as ica_epo.fif. 
    This placeholder represents where additional specific formatting would be applied
    if starting from absolute raw files.
    """
    print("\n--- Nguyen Dataset Formatting ---")
    print("The Nguyen dataset is assumed to be pre-formatted as ica_epo.fif.")
    # In practice, this would involve loading .mat or raw files, applying bandpass (4-80Hz)
    # and ICA blink removal as described in the paper.

if __name__ == "__main__":
    format_bcicomp()
    format_nguyen()
