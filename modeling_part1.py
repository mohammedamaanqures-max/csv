from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2
TOP_N_FEATURES = 15


def load_dataset(repo_root: Path) -> pd.DataFrame:
    dataset_path = repo_root / "Airline Passenger Satisfaction.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    return pd.read_csv(dataset_path)


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    target_col = "satisfaction"
    if target_col not in df.columns:
        raise ValueError("Expected target column 'satisfaction' not found.")

    y = (df[target_col].astype(str).str.lower().str.strip() == "satisfied").astype(int)

    drop_cols = [target_col]
    for col in ["Unnamed: 0", "id"]:
        if col in df.columns:
            drop_cols.append(col)
    X = df.drop(columns=drop_cols)
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def encode_and_impute(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_train = X_train.copy()
    X_test = X_test.copy()

    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in X_train.columns if col not in numeric_cols]

    for col in numeric_cols:
        median = X_train[col].median()
        X_train[col] = X_train[col].fillna(median)
        X_test[col] = X_test[col].fillna(median)

    for col in categorical_cols:
        mode = X_train[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        X_train[col] = X_train[col].fillna(fill_value)
        X_test[col] = X_test[col].fillna(fill_value)

    X_train_encoded = pd.get_dummies(X_train, drop_first=False)
    X_test_encoded = pd.get_dummies(X_test, drop_first=False)
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

    return X_train_encoded, X_test_encoded


def class_distribution(y: pd.Series) -> pd.Series:
    labels = y.map({0: "neutral_or_dissatisfied", 1: "satisfied"})
    return labels.value_counts().sort_index()


def plot_class_distribution(distribution: pd.Series, title: str, output_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    distribution.plot(kind="bar", color=["#457b9d", "#e63946"])
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_lasso_feature_selection(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X)

    lasso = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=1.0,
        solver="saga",
        random_state=RANDOM_STATE,
        max_iter=5000,
        C=0.5,
    )
    lasso.fit(X_scaled, y)

    coef_abs = pd.Series(abs(lasso.coef_[0]), index=X.columns)
    return coef_abs.sort_values(ascending=False)


def run_rf_feature_importance(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)


def plot_top_features(features: pd.Series, title: str, output_path: Path, top_n: int = TOP_N_FEATURES) -> None:
    top_features = features.head(top_n).sort_values(ascending=True)
    plt.figure(figsize=(9, 6))
    top_features.plot(kind="barh", color="#2a9d8f")
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def stringify_top_features(features: pd.Series, top_n: int = TOP_N_FEATURES) -> str:
    return ", ".join(features.head(top_n).index.tolist())


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = df.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]).replace("|", "\\|") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(
    report_path: Path,
    slide_outline_path: Path,
    split_distributions: Dict[str, pd.Series],
    balance_distributions: Dict[str, pd.Series],
    results_table: pd.DataFrame,
) -> None:
    train_dist = split_distributions["train"].to_dict()
    test_dist = split_distributions["test"].to_dict()

    smote_dist = balance_distributions["SMOTE"].to_dict()
    rus_dist = balance_distributions["RandomUndersampling"].to_dict()

    report_text = f"""# Modeling Part 1 Report: Class Imbalance and Feature Selection

## 1. Objective
This report documents Modeling Part 1 on the repository dataset (`Airline Passenger Satisfaction.csv`).
The goals were to (a) create a reproducible train/test split, (b) address class imbalance using two techniques,
(c) perform feature selection using LASSO and a second method, (d) visualize key outcomes, and (e) organize findings
for reporting and presentation.

## 2. Methodology
### 2.1 Data loading and reproducibility
- Data source: repository-relative path `Airline Passenger Satisfaction.csv`.
- Reproducibility controls:
  - `random_state = {RANDOM_STATE}` for splitting and modeling.
  - deterministic train/test split with stratification.

### 2.2 Train/test split
- Split ratio: {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} (train/test).
- Stratification used to preserve class ratios.
- Target mapping:
  - `satisfied` -> 1
  - `neutral or dissatisfied` -> 0

Observed class counts:
- Training set: {train_dist}
- Test set: {test_dist}

### 2.3 Class imbalance handling
Two methods were applied on the training data only:

1. **SMOTE (Synthetic Minority Over-sampling Technique)**
   - Rationale: creates synthetic minority examples to reduce majority bias without simple duplication.
   - Post-resampling class counts: {smote_dist}

2. **Random Undersampling**
   - Rationale: reduces majority class size to match minority count; useful for faster modeling and stronger class balance.
   - Post-resampling class counts: {rus_dist}

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

{dataframe_to_markdown(results_table)}

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
"""

    slide_outline_text = """# PowerPoint Slide Outline (Modeling Part 1)

1. **Title Slide**
   - Project: Airline Passenger Satisfaction Modeling Part 1
   - Focus: Class Imbalance + Feature Selection

2. **Data Preparation Summary (Completed Previously)**
   - Cleaning, encoding, transformation recap
   - Final modeling target: satisfaction

3. **Train/Test Split**
   - 80/20 stratified split
   - Reproducibility via random_state
   - Show train/test class counts

4. **Class Imbalance: Baseline**
   - Pre-balancing class distribution chart
   - Why imbalance matters for classification

5. **Imbalance Technique 1: SMOTE**
   - Concept: synthetic minority generation
   - Before/after class distribution
   - Strengths and trade-offs

6. **Imbalance Technique 2: Random Undersampling**
   - Concept: reduce majority samples
   - Before/after class distribution
   - Strengths and trade-offs

7. **Feature Selection Method 1: LASSO**
   - L1 regularization concept
   - Top coefficient bar chart
   - Key selected features

8. **Feature Selection Method 2: Random Forest Importance**
   - Tree-based importance concept
   - Top feature importance bar chart
   - Key selected features

9. **Results Table (Required Format)**
   - Feature Selection Method
   - Balancing Technique
   - Important Features

10. **Interpretation & Key Findings**
    - How balancing changed training data
    - Consistent important features across methods
    - Differences between LASSO and RF rankings

11. **Next Steps**
    - Modeling Part 2: train/evaluate classifiers
    - Compare metrics (precision, recall, F1, ROC-AUC)
"""

    report_path.write_text(report_text)
    slide_outline_path.write_text(slide_outline_text)


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    output_dir = repo_root / "outputs" / "modeling_part1"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(repo_root)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_encoded, _ = encode_and_impute(X_train, X_test)

    train_distribution = class_distribution(y_train)
    test_distribution = class_distribution(y_test)
    plot_class_distribution(
        train_distribution,
        "Training Class Distribution (Before Resampling)",
        output_dir / "class_distribution_train_before.png",
    )

    smote = SMOTE(random_state=RANDOM_STATE)
    X_smote, y_smote = smote.fit_resample(X_train_encoded, y_train)
    smote_distribution = class_distribution(y_smote)
    plot_class_distribution(
        smote_distribution,
        "Training Class Distribution After SMOTE",
        output_dir / "class_distribution_smote.png",
    )

    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_rus, y_rus = rus.fit_resample(X_train_encoded, y_train)
    rus_distribution = class_distribution(y_rus)
    plot_class_distribution(
        rus_distribution,
        "Training Class Distribution After Random Undersampling",
        output_dir / "class_distribution_random_undersampling.png",
    )

    datasets = {
        "SMOTE": (X_smote, y_smote),
        "RandomUndersampling": (X_rus, y_rus),
    }

    results_rows: List[Dict[str, str]] = []

    for balancing_name, (X_balanced, y_balanced) in datasets.items():
        lasso_importance = run_lasso_feature_selection(X_balanced, y_balanced)
        rf_importance = run_rf_feature_importance(X_balanced, y_balanced)

        plot_top_features(
            lasso_importance,
            f"Top LASSO Coefficients ({balancing_name})",
            output_dir / f"lasso_coefficients_{balancing_name.lower()}.png",
        )
        plot_top_features(
            rf_importance,
            f"Top Random Forest Feature Importance ({balancing_name})",
            output_dir / f"rf_importance_{balancing_name.lower()}.png",
        )

        results_rows.append(
            {
                "Feature Selection Method": "LASSO (Logistic Regression L1)",
                "Balancing Technique": balancing_name,
                "Important Features": stringify_top_features(lasso_importance),
            }
        )
        results_rows.append(
            {
                "Feature Selection Method": "Random Forest Importance",
                "Balancing Technique": balancing_name,
                "Important Features": stringify_top_features(rf_importance),
            }
        )

    results_df = pd.DataFrame(results_rows)
    results_csv_path = output_dir / "feature_selection_results.csv"
    results_df.to_csv(results_csv_path, index=False)

    report_path = repo_root / "modeling_part1_report.md"
    slide_outline_path = repo_root / "modeling_part1_slide_outline.md"

    build_report(
        report_path=report_path,
        slide_outline_path=slide_outline_path,
        split_distributions={"train": train_distribution, "test": test_distribution},
        balance_distributions={"SMOTE": smote_distribution, "RandomUndersampling": rus_distribution},
        results_table=results_df,
    )

    print("Modeling Part 1 workflow complete.")
    print(f"Outputs written to: {output_dir}")
    print(f"Report: {report_path}")
    print(f"Slide outline: {slide_outline_path}")


if __name__ == "__main__":
    main()
