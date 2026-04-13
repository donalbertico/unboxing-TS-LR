# Robust BCI Classification: Spatial & Spectral Compression Analysis

This repository contains the processing pipeline and analysis scripts for evaluating data compression and frequency invariance in EEG-based Brain-Computer Interfaces (BCI). The study focuses on binary classification tasks across two distinct datasets using Riemannian geometry and Tangent Space projection.

## 📊 Results Summary

The analysis produced three primary compound visualizations:
1. **Methodology Comparison**: A statistical validation of Cross-Validation strategies (Group vs. Stratified) and Sanity Checks, featuring binomial significance thresholds (p < 0.01).
2. **Invariance Test**: A dual-panel figure showcasing the hyper-compression landscape (Channels vs. Frequency Bands) and the robustness of classification across different neurophysiological filter sets.
3. **Compound Spatial-Spectral Analysis**: A deep dive into the top performers, combining topographic importance maps with frequency-band correlation heatmaps.

## 📂 Repository Structure

- `run_on_val_dual_optimized.py`: Optimized hyperparameter sweep for the BCI Competition dataset.
- `nguyen_cv_dual_optimized.py`: Optimized hyperparameter sweep for the Nguyen dataset.
- `run_final_filter_sweep_bci.py`: Verification of specific frequency filter sets for BCIComp.
- `run_final_filter_sweep_nguyen.py`: Verification of specific frequency filter sets for Nguyen.
- `generate_methodology_v2.py`: Generates the methodology validation boxplots with significance thresholds.
- `generate_invariance_test.py`: Generates the compression grids and filter invariance boxplots.
- `generate_compound_analysis.py`: Generates the spatial importance and band correlation figure.
- `figures/`: Contains the high-resolution generated plots.

## 🛰️ Data Sources

This repository **does not contain the raw EEG data**. To replicate the results, please download the datasets from the following sources:

1. **BCI Competition Dataset (Track 3)**:
   - **Task**: Binary classification of "Stop" vs. "Thank You" imagery.
   - **Source**: [Download from Dropbox](https://www.dropbox.com/scl/fi/20j120qae7c2rlmr5lfwr/Dataset.zip?rlkey=0xjdairhprrakmw27d2fnesj7&e=1&dl=0)
   - **Processing**: The scripts expect `gemini_ica_epo.fif` and `gemini_validation_ica.fif` files in subject-specific folders (`sXX`).

2. **Nguyen et al. (2018) Dataset**:
   - **Task**: Binary classification of "Short" vs. "Long" word imagery.
   - **Source**: [View on OSF](https://osf.io/pq7vb/overview)
   - **Processing**: The scripts expect `ica_epo.fif` files in subject-specific folders (`sub_XX`).

## 🛠️ Methodology

The pipeline follows a rigorous Riemannian approach:
- **Preprocessing**: 1-127Hz (BCI) / 1-80Hz (Nguyen) bandpass filtering and ICA-based artifact removal.
- **Covariance Estimation**: Ledoit-Wolf Shrinkage (LWF) estimator.
- **Tangent Space Projection**: Mapping covariance matrices to a Euclidean space while preserving the Riemannian manifold structure.
- **Classification**: L1-regularized Logistic Regression (SAGA solver) for sparse feature selection.
- **Dual Selection**: Subject-specific ranking of channel and frequency band importance within each CV fold to evaluate performance under extreme data compression.
- **Validation**: 10-fold GroupKFold cross-validation to prevent trial/block leakage.

## 🚀 Usage

1. **Install Dependencies**:
   ```bash
   pip install mne pyriemann scikit-learn numpy matplotlib seaborn scipy
   ```
2. **Run Analysis**:
   ```bash
   python run_optimized_parallel.py  # Runs the full hyperparameter sweep
   python run_final_sweeps.py        # Runs the filter set verification
   ```
3. **Generate Plots**:
   ```bash
   python plot_methodology_comparison_v2.py
   python generate_invariance_test.py
   python generate_compound_analysis.py
   ```

---
*Created with love for BCI research. xoxo*
