# PowerPoint Outline: Modeling Part 1

## Slide 1 — Title
- Modeling Part 1: Class Imbalance Handling and Feature Selection
- Course / Project name
- Student name and date

## Slide 2 — Dataset & Data Preparation Summary
- Dataset name and size
- Target variable
- Data preparation already completed (cleaning, missing values, encoding, transformations)
- Final modeling-ready feature count

## Slide 3 — Train/Test Split Strategy
- Split ratio used (e.g., 80/20)
- Why stratification was used for classification
- Reproducibility with random state
- Train vs test sample counts

## Slide 4 — Why Class Imbalance Matters
- Brief definition of class imbalance
- Risks if left untreated (bias toward majority class)
- Goal for this phase: fairer learning signal

## Slide 5 — Balancing Technique 1: SMOTE
- How SMOTE works (synthetic minority samples)
- When it is useful
- Before/after class distribution table or chart

## Slide 6 — Balancing Technique 2: RandomUnderSampler
- How random undersampling works
- Benefits and trade-offs
- Before/after class distribution table or chart

## Slide 7 — Feature Selection Method 1: LASSO
- L1 regularization concept
- Why used in this workflow
- Top features discovered
- Insert LASSO coefficient bar chart

## Slide 8 — Feature Selection Method 2: Random Forest Importance
- Tree-based importance concept
- Why complementary to LASSO
- Top features discovered
- Insert feature-importance bar chart

## Slide 9 — Combined Results Table
- Show table with columns:
  - Feature Selection Method
  - Balancing Technique
  - Important Features
- Highlight overlapping important features

## Slide 10 — Interpretation & Key Findings
- Effect of balancing on data representation
- Most influential features and domain intuition
- Differences between methods
- Practical implications for next modeling phase

## Slide 11 — Limitations & Next Steps
- Limitations (e.g., feature importance stability, metric dependence)
- Next step: model benchmarking and hyperparameter tuning
- Add evaluation metrics plan (accuracy/F1/ROC-AUC as applicable)

## Slide 12 — Q&A
- Backup slide references
- Additional plots/tables in appendix

