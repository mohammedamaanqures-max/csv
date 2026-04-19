# Modeling Part 1 Report: Class Imbalance Handling and Feature Selection

## 1. Executive Summary

This report documents a complete and reproducible **Modeling Part 1** workflow for a prepared dataset. The objective is to build a transparent modeling pipeline that (1) performs a reproducible train/test split, (2) addresses class imbalance, (3) identifies important features using multiple selection methods, and (4) organizes all outcomes into reusable output artifacts for analysis and presentation.

The workflow is implemented in `modeling_part1.py` and is designed to be configurable via command-line arguments so it can be reused on other prepared datasets with minimal effort. The implementation uses standard scientific Python libraries: **pandas**, **scikit-learn**, **imbalanced-learn**, **matplotlib**, and **seaborn**.

### Key goals of this phase
- Create a reliable baseline process before advanced model tuning.
- Quantify and document how balancing methods affect class counts.
- Compare feature selection methods to understand which predictors are consistently influential.
- Produce saved outputs that can be directly included in reports and slide decks.

---

## 2. Methodology

### 2.1 Input assumptions
This phase assumes the dataset has already completed data preparation (cleaning, missing value handling, encoding, and transformations). The script still includes robust checks such as dropping unnamed index columns and one-hot encoding any remaining non-numeric fields for safety.

### 2.2 Configurable inputs
The script supports the following inputs:
- `--data-path`: path to prepared CSV dataset
- `--target-column`: target variable name
- `--test-size`: train/test split ratio (default `0.2` = 80/20)
- `--random-state`: random seed for reproducibility (default `42`)
- `--top-n`: number of top features to report/plot (default `10`)
- `--balancing-techniques`: comma-separated methods (default `smote,random_under`)
- `--outputs-dir`: output folder for tables and plots (default `outputs`)

### 2.3 Reproducible train/test split
The workflow performs train/test split using `train_test_split`.
- Default split: **80/20**
- For classification targets: stratification is applied to preserve class proportions.
- `random_state` ensures exact reproducibility across runs.

### 2.4 Class imbalance handling
Two balancing methods are included by default, and additional options are available.

#### A) SMOTE (Synthetic Minority Oversampling Technique)
- Generates synthetic minority samples based on nearest neighbors.
- Useful when minority class is underrepresented and retaining all majority data is beneficial.
- Helps models learn smoother minority boundaries without simple duplication.

#### B) RandomUnderSampler
- Randomly removes samples from majority class.
- Useful when dataset is large and majority class dominates learning.
- Fast and simple, but may discard informative majority examples.

#### Optional methods
- **ADASYN**: adaptive synthetic oversampling (focuses more on difficult minority regions)
- **RandomOverSampler**: direct minority duplication

For each selected technique, the script logs and saves class distributions:
- Before resampling
- After resampling

These results are written to `outputs/class_distribution_log.csv`.

### 2.5 Feature selection methods
Two feature selection methods are applied for each balancing technique.

#### A) LASSO (mandatory)
- For classification: `LogisticRegression` with **L1** penalty.
- For regression targets: `Lasso` regression.
- L1 regularization pushes less informative coefficients toward zero.
- Output: absolute coefficient magnitudes ranked by importance.

#### B) Random Forest feature importance
- Uses ensemble tree-based model importance scores.
- Captures non-linear relationships and interactions.
- Output: impurity-based feature importance ranking.

### 2.6 Leakage prevention and robust modeling flow
Resampling is applied **only on training data** through `imblearn` pipelines, preventing test leakage. The test split remains untouched for future evaluation stages.

### 2.7 Visual outputs
For each balancing technique, the script saves:
- LASSO coefficient bar chart (`lasso_coefficients_<technique>.png`)
- Random Forest importance bar chart (`rf_importance_<technique>.png`)

### 2.8 Results table structure
All combinations are consolidated in:
`outputs/modeling_part1_results.csv`

Columns:
- **Feature Selection Method**
- **Balancing Technique**
- **Important Features**

---

## 3. Interpretation Framework (Dataset-Specific)

Use this section to interpret your actual run outputs.

### 3.1 Impact of imbalance handling
Replace placeholders with observed values from `class_distribution_log.csv`.

- Original training distribution: `[CLASS_A: ___, CLASS_B: ___, ...]`
- After SMOTE: `[CLASS_A: ___, CLASS_B: ___, ...]`
- After RandomUnderSampler: `[CLASS_A: ___, CLASS_B: ___, ...]`

Interpretation guidance:
1. If SMOTE creates a near-balanced dataset, minority recall often improves because the learner sees more minority patterns.
2. If RandomUnderSampler reduces majority size substantially, training time may improve, but information loss may occur.
3. If both methods produce different feature rankings, that indicates class-balance strategy is influencing model focus.

### 3.2 Most important features and why
Extract top features from `modeling_part1_results.csv` and fill:

- LASSO + SMOTE top features: `[f1, f2, f3, ...]`
- RandomForest + SMOTE top features: `[f1, f2, f3, ...]`
- LASSO + RandomUnder top features: `[f1, f2, f3, ...]`
- RandomForest + RandomUnder top features: `[f1, f2, f3, ...]`

Interpretation guidance:
- Features appearing across multiple methods are strong candidates for robust signal.
- LASSO highlights linear sparse effects.
- Random Forest can highlight interaction/nonlinear effects that LASSO may underweight.

### 3.3 Differences between feature selection methods
Typical observations to report:
- **LASSO** may rank fewer, more stable linear predictors.
- **Random Forest** may distribute importance across correlated predictors.
- If rankings overlap highly, confidence in feature relevance increases.
- If rankings diverge strongly, investigate feature interactions, multicollinearity, and class-balance sensitivity.

---

## 4. Reproducibility Checklist

- [x] Configurable dataset path and target column
- [x] Reproducible split with fixed random state
- [x] Stratified split for classification
- [x] Multiple imbalance methods with before/after logging
- [x] Two feature selection methods (LASSO + Random Forest)
- [x] Saved plots and CSV outputs
- [x] Methodology and interpretation guidance documented

---

## 5. Key Findings Template (Fill After Run)

Use this short template in your final submission:

1. **Imbalance impact**: "Using [SMOTE/RandomUnderSampler], minority class representation changed from [__] to [__], which is expected to [improve/decrease] model sensitivity to minority outcomes."
2. **Feature insights**: "The most consistently important features were [__], appearing across [__] methods and balancing settings."
3. **Method differences**: "LASSO emphasized [__], while Random Forest emphasized [__], suggesting [linear/nonlinear] structure in the data."
4. **Practical recommendation**: "For the next modeling phase, prioritize [balancing method] with [feature subset strategy] and validate through cross-validated predictive metrics."

---

## 6. Conclusion

This Modeling Part 1 workflow establishes a reproducible foundation for supervised modeling by integrating data splitting discipline, imbalance handling, and feature discovery into one script-driven process. The generated outputs are intentionally report-friendly and presentation-ready. The process can be rerun with different targets, split ratios, and balancing methods while preserving methodological consistency and reproducibility.

