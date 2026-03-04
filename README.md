# ML-Poli-Sci

Code and simulations for machine-learning–based voting analysis and domain adaptation in political science.

## Reproducing the paper results

### 1. Environment

- **Python version**: 3.8+ (the original project used Anaconda with Python 3.6–3.9)
- **Core packages**:
  - `numpy`, `pandas`, `matplotlib`
  - `scipy`, `scikit-learn`
  - (optional) `imbalanced-learn` for some imbalance-handling experiments

A minimal installation (from a clean environment) could be:

```bash
pip install numpy pandas matplotlib scipy scikit-learn
```

### 2. Code structure (high level)

- **Top level**
  - `simulate_tca.py`  
    Monte Carlo simulation for the TCA domain-adaptation study (paper Section 5).  
    Reproduces Tables 1–4 and Figures 1–3 for the synthetic experiment.

- **`code/`**
  - `utils.py`  
    Core utilities for data loading, missing-value analysis, feature filtering, feature processing, and the `universal_predict` / `universal_predict_TCA` pipelines used in the empirical analyses.
  - `PCA_TCA_run.py`  
    Script that runs the main empirical prediction experiment on the cumulative survey data and generates PCA/TCA visualisations for voters vs. non-voters.
  - `transfer_learn.py`  
    Implementation of the TCA (Transfer Component Analysis) method and related helpers for domain adaptation.
  - `universal_prediction.ipynb`  
    Jupyter notebook used for interactive exploration and checking of the universal-prediction pipeline and feature choices.

- **`data/`**
  - `cumulative_2022_v3_9_domain.csv` (and related `*.npy` mapping files) – processed cumulative survey with domain variables.
  - `universal_predict/...` – output folders for filtered feature lists, prediction results, and figures (e.g., PCA/TCA plots).

### 3. How to run the main experiments

#### 3.1 Simulation: TCA Monte Carlo study (paper Section 5)

- **From the repository root**:

```bash
python simulate_tca.py
```

- **Outputs** (saved in the same folder as `simulate_tca.py`):
  - `simulation_results.csv` – raw condition-level results.
  - `table1_rmse.csv` – Table 1 (RMSE).
  - `table2_bias.csv` – Table 2 (prediction bias, \(n_s = 1000\)).
  - `table3_mmd.csv` – Table 3 (MMD before/after TCA).
  - `table4_relative.csv` – Table 4 (relative improvement of TCA+Elastic-Net vs. Elastic-Net).
  - `fig1_rmse_by_shift.png` – Figure 1 (RMSE vs. shift severity).
  - `fig2_mmd_reduction.png` – Figure 2 (MMD before/after TCA).
  - `fig3_bias.png` – Figure 3 (mean prediction bias by method and shift).

These files correspond directly to the tables and figures reported in the simulation section of the paper.

#### 3.2 Empirical voting prediction and TCA visualisation

The main empirical pipeline lives in `code/utils.py` and is invoked by `code/PCA_TCA_run.py`.

- **Data prerequisites**:
  - `data/cumulative_2022_v3_9_domain.csv`
  - `data/column_to_variable_dict.npy`
  - `data/variable_to_column_dict.npy`
  - `data/value_labels.npy`

  These files are already expected by `PCA_TCA_run.py` at the relative paths used in the script.

- **To run the empirical experiment** (from the repository root):

```bash
cd code
python PCA_TCA_run.py
```

- **What this script does (high level)**:
  - Filters training and test sets based on voting-related variables (e.g., `Voted_D_R`, `Voted`).
  - Performs missing-value analysis and feature filtering (thresholds on recent-year missingness).
  - Determines numerical vs. categorical features and builds the feature lists.
  - Runs the universal prediction pipeline (baseline model) and the TCA-augmented pipeline.
  - Produces PCA plots comparing:
    - TCA-transformed features (`PCA_TCA_diff.png`), and
    - Raw features (`PCA_raw_diff.png`)
    in the corresponding `../data/universal_predict/...` output folder, stratified by vote outcome (D/R) and non-voters.

These figures and prediction outputs are those referenced when discussing empirical results in the paper (state-wise and overall voting behaviour).

### 4. Mapping scripts to paper sections

- **Simulation section (e.g., Section 5)**:
  - `simulate_tca.py` – reproduces all simulation tables and figures.

- **Empirical section (state-wise and overall predictions)**:
  - `code/utils.py` – missing-data handling, feature engineering, and predictive models.
  - `code/PCA_TCA_run.py` – end-to-end script for the empirical TCA experiment and PCA visualisations.

You can cite/describe the code in the paper by referencing these scripts and their outputs.

> This README is anonymised for peer review. Additional project history,
> external links, and provenance information are intentionally omitted and
> can be provided after the review process.
