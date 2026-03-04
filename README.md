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

## Resources

Slides: https://docs.google.com/presentation/d/1Qzgl6P9cNUejpLUx-ivtGDouEj2i82tKl3jF-sBwdbU/edit?usp=sharing

Missing value imputation-R: https://github.com/IQSS/amelia & https://gking.harvard.edu/amelia

Missing value imputation-sklearn:https://scikit-learn.org/stable/modules/impute.html

## log

2024/6/11
- add justifications for the threshold of the missing value ratio

- 2024/4/18-5/30:
   - add variables: VCF0006a” or “Unique Respondent Number (Cross-year ID for panel cases), voter/non-voter, vote_D/vote_R to the state-wise prediction results:
        - VCF0006a: done
        - voter/non-voter,vote_D/vote_R: already in the data

   - add the one model trained on the whole data and the state-wise model prediction results:
        - to do in - done

   - add the citation of the ML term in documents
        - done
   - clean the code, make it:
        - script-run for new data test (apply curent model to new data) - doing
        - script-run for new data training (model update)
        - visulization and store results script -done


- 2024/4/11:
   - send documents on lgistic regression + elastic net to the professor
   - make the documetns on data process ( one-hot  ) and missing value imputation
   - build a table to show the gap between the "vote-D" and "vote-R" for whole data and state-wise data


- 2024/3/26:
    - model did not work well on "intend-vote" group due to the imbalance of the data
    - try some imbalanced data handling methods, like SMOTE, ADASYN, and RandomOverSampler, but did not work well
    - tried some advanced/non-linear models, like GBTtree, RBF-SVM, and ensemble models, like AdaBoost , but did not work well
    - clean the code and add some comments


- 2024/3/12:
    - save and finished almost all stat-based analysis
    - start to build the feature-importance model (Log-Reg)
    - focus on the "WA" state group


- 2024/2/23:
 - add table for "intend to vote" but final "non-voter" for the white/black group in different area
 - add year-based plotting for the changing ratio

- get some  hypotheses to verify(based on "urban-rural"-feature, not miss out!): 

    - Blacks in urban America are more likely to vote and vote for the Democratic candidates than are Blacks in rural America.
 
    - Blacks in suburban America are more likely to vote for the Republican candidates than are Blacks in urban America.
 
    - Whites in rural America are more likely to vote Republicans than are Whites in urban or suburban America.
 
    - White non-voters are more likely to live in rural American than in urban America.

- state-based analysis: focus on WA

- foucus on final non-voters who intend to vote

- after feature filter-out, check the performance with only top-5/10/20 features 




2024 Jan 3rd: 

- finish the feature filtering: set the missing-ratio  in recent 20 years as the threshold, and remove the features with missing ratio larger than 0.3-done

- build the simple classifier: logistic regression, random forest, and gradient boosting-doing

### to-do:
- using the small dataset (no missing values) to build the simple classifier
- send email to the professor to ask about the missing values and categorical features
- using the sklearn/R based imputation method to deal with the missing values 


### problem: 
how to deal with the missing data?

- just drop? -> only 10% of the data left( 6000~7000 samples)
- fill with mean? -> too much categorical data
- use the `amelia` package? -> can it deal with the categorical data? -> to check
- use the `sklearn.impute` package? -> can it deal with the categorical data? -> to check
- use 

how to deal with the categorical features?
- should we use the label "UK/don't want answer" as a category, or just drop it/as a missing values?
-  is so, can we use the `sklearn.preprocessing.OneHotEncoder` to encode the categorical features? -> to check

2023 Dec: 

- finish the data collection and process, set the target - Done

