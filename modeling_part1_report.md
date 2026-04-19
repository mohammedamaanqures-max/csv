# Modeling Part 1 Report: Class Imbalance and Feature Selection

## 1. Objective
This report documents Modeling Part 1 on the repository dataset (`Airline Passenger Satisfaction.csv`).
The goals were to (a) create a reproducible train/test split, (b) address class imbalance using two techniques,
(c) perform feature selection using LASSO and a second method, (d) visualize key outcomes, and (e) organize findings
for reporting and presentation.

## 2. Methodology
### 2.1 Data loading and reproducibility
- Data source: repository-relative path `Airline Passenger Satisfaction.csv`.
- Reproducibility controls:
  - `random_state = 42` for splitting and modeling.
  - deterministic train/test split with stratification.

### 2.2 Train/test split
- Split ratio: 80/20 (train/test).
- Stratification used to preserve class ratios.
- Target mapping:
  - `satisfied` -> 1
  - `neutral or dissatisfied` -> 0

Observed class counts:
- Training set: {'neutral_or_dissatisfied': 47103, 'satisfied': 36020}
- Test set: {'neutral_or_dissatisfied': 11776, 'satisfied': 9005}

### 2.3 Class imbalance handling
Two methods were applied on the training data only:

1. **SMOTE (Synthetic Minority Over-sampling Technique)**
   - Rationale: creates synthetic minority examples to reduce majority bias without simple duplication.
   - Post-resampling class counts: {'neutral_or_dissatisfied': 47103, 'satisfied': 47103}

2. **Random Undersampling**
   - Rationale: reduces majority class size to match minority count; useful for faster modeling and stronger class balance.
   - Post-resampling class counts: {'neutral_or_dissatisfied': 36020, 'satisfied': 36020}

Interpretation:
- SMOTE retained all original majority records while increasing minority examples.
- Random undersampling balanced classes by discarding majority examples.
- In practice, SMOTE often preserves information better, while undersampling can train faster but may remove signal.

### 2.4 Feature selection techniques
Two feature selection methods were used for each balancing approach:

1. **LASSO (L1-regularized Logistic Regression)**
   - Features with larger absolute coefficients are treated as more influential.
   - Encourages sparse solutions and interpretability.

2. **Random Forest Feature Importance**
   - Measures each feature's contribution via impurity reduction across trees.
   - Captures non-linear effects and interactions.

### 2.5 Visualizations generated
- Class distribution charts:
  - `outputs/modeling_part1/class_distribution_train_before.png`
  - `outputs/modeling_part1/class_distribution_smote.png`
  - `outputs/modeling_part1/class_distribution_random_undersampling.png`
- Feature charts:
  - LASSO and Random Forest importance plots for each balancing method in `outputs/modeling_part1/`.

## 3. Results table
The table below reports the requested structure (Feature Selection Method, Balancing Technique, Important Features).

| Feature Selection Method | Balancing Technique | Important Features |
| --- | --- | --- |
| LASSO (Logistic Regression L1) | SMOTE | Gender_Male, Gender_Female, Class_Business, Class_Eco, Type of Travel_Business travel, Customer Type_Loyal Customer, Customer Type_disloyal Customer, Class_Eco Plus, Type of Travel_Personal Travel, Online boarding, Inflight wifi service, Checkin service, On-board service, Arrival Delay in Minutes, Leg room service |
| Random Forest Importance | SMOTE | Online boarding, Inflight wifi service, Class_Business, Type of Travel_Business travel, Inflight entertainment, Type of Travel_Personal Travel, Seat comfort, Ease of Online booking, Customer Type_Loyal Customer, Flight Distance, Age, Class_Eco, Leg room service, On-board service, Baggage handling |
| LASSO (Logistic Regression L1) | RandomUndersampling | Type of Travel_Personal Travel, Online boarding, Customer Type_disloyal Customer, Inflight wifi service, Checkin service, Class_Eco, On-board service, Arrival Delay in Minutes, Leg room service, Cleanliness, Ease of Online booking, Class_Eco Plus, Departure/Arrival time convenient, Departure Delay in Minutes, Baggage handling |
| Random Forest Importance | RandomUndersampling | Online boarding, Inflight wifi service, Class_Business, Type of Travel_Personal Travel, Inflight entertainment, Type of Travel_Business travel, Seat comfort, Ease of Online booking, Class_Eco, Flight Distance, On-board service, Age, Leg room service, Customer Type_disloyal Customer, Customer Type_Loyal Customer |

## 4. Interpretation of findings
- **Effect of imbalance handling:** both methods produced balanced training distributions; however, they do so differently.
  SMOTE increases minority representation synthetically, while undersampling shrinks majority representation.
- **Most important features:** top-ranked features repeatedly include service-quality and delay-related variables,
  indicating customer experience and operational reliability are strong drivers of satisfaction labels.
- **Differences between selection methods:**
  - LASSO emphasizes linearly separable predictors and suppresses weak/redundant variables.
  - Random Forest can emphasize variables that interact non-linearly and may distribute importance across related features.

## 5. Conclusion
This Modeling Part 1 workflow is reproducible, repository-path based, and aligns with assignment requirements:
train/test split, class imbalance handling with two methods, feature selection via LASSO + Random Forest,
visualizations, and a consolidated results table. The outputs support downstream model comparison in later phases.
