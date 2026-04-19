import argparse
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MAX_UNIQUE_FOR_CLASSIFICATION = 20
MAX_UNIQUE_RATIO_FOR_CLASSIFICATION = 0.05
DEFAULT_LASSO_ALPHA = 0.01
DEFAULT_RF_ESTIMATORS = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modeling Part 1: class balancing + feature selection workflow"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="Airline Passenger Satisfaction.csv",
        help="Path to prepared dataset file (CSV)",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        required=True,
        help="Target column name",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top N features to report and visualize",
    )
    parser.add_argument(
        "--balancing-techniques",
        type=str,
        default="smote,random_under",
        help=(
            "Comma-separated balancing methods: "
            "smote,random_under,adasyn,random_over"
        ),
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Directory to store outputs",
    )
    return parser.parse_args()


def infer_problem_type(y: pd.Series) -> str:
    if y.dtype.name in {"object", "category", "bool"}:
        return "classification"

    unique_count = y.nunique(dropna=False)
    unique_ratio = unique_count / max(len(y), 1)
    if (
        unique_count <= MAX_UNIQUE_FOR_CLASSIFICATION
        or unique_ratio <= MAX_UNIQUE_RATIO_FOR_CLASSIFICATION
    ):
        return "classification"
    return "regression"


def load_dataset(data_path: str, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_path)
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed") or str(col) == ""]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset columns.")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Ensure any remaining categorical/object columns become numeric model inputs.
    X = pd.get_dummies(X, drop_first=False)

    return X, y


def class_distribution(y: pd.Series) -> pd.Series:
    return y.value_counts(dropna=False).sort_index()


def plot_top_features(
    feature_df: pd.DataFrame,
    feature_col: str,
    score_col: str,
    title: str,
    output_path: Path,
    top_n: int,
) -> None:
    top_df = feature_df.head(top_n).copy()

    plt.figure(figsize=(10, max(4, int(top_n * 0.45))))
    sns.barplot(data=top_df, x=score_col, y=feature_col, color="#4C78A8")
    plt.title(title)
    plt.xlabel(score_col)
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def lasso_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str,
    resampler,
    random_state: int,
) -> pd.DataFrame:
    if problem_type == "classification":
        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            random_state=random_state,
            max_iter=5000,
        )
        pipeline = ImbPipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("resampler", resampler),
                ("model", model),
            ]
        )
    else:
        model = Lasso(
            alpha=DEFAULT_LASSO_ALPHA, random_state=random_state, max_iter=10000
        )
        pipeline = ImbPipeline(steps=[("scaler", StandardScaler()), ("model", model)])

    pipeline.fit(X_train, y_train)
    fitted_model = pipeline.named_steps["model"]

    if problem_type == "classification" and getattr(fitted_model.coef_, "ndim", 1) > 1:
        coef_values = fitted_model.coef_
        coef_abs = abs(coef_values).mean(axis=0)
    else:
        coef_abs = abs(fitted_model.coef_)

    feature_df = (
        pd.DataFrame({"Feature": X_train.columns, "LASSO_AbsCoefficient": coef_abs})
        .sort_values("LASSO_AbsCoefficient", ascending=False)
        .reset_index(drop=True)
    )
    return feature_df


def rf_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str,
    resampler,
    random_state: int,
) -> pd.DataFrame:
    if problem_type == "classification":
        model = RandomForestClassifier(
            n_estimators=DEFAULT_RF_ESTIMATORS,
            random_state=random_state,
            n_jobs=-1,
        )
        pipeline = ImbPipeline(steps=[("resampler", resampler), ("model", model)])
    else:
        model = RandomForestRegressor(
            n_estimators=DEFAULT_RF_ESTIMATORS,
            random_state=random_state,
            n_jobs=-1,
        )
        pipeline = ImbPipeline(steps=[("model", model)])

    pipeline.fit(X_train, y_train)
    importances = pipeline.named_steps["model"].feature_importances_

    feature_df = (
        pd.DataFrame({"Feature": X_train.columns, "RF_Importance": importances})
        .sort_values("RF_Importance", ascending=False)
        .reset_index(drop=True)
    )
    return feature_df


def get_balancer_builders(method_names: List[str]) -> Dict[str, Callable[[int], object]]:
    builders: Dict[str, Callable[[int], object]] = {
        "smote": lambda rs: SMOTE(random_state=rs),
        "random_under": lambda rs: RandomUnderSampler(random_state=rs),
        "adasyn": lambda rs: ADASYN(random_state=rs),
        "random_over": lambda rs: RandomOverSampler(random_state=rs),
    }
    selected: Dict[str, Callable[[int], object]] = {}
    for method in method_names:
        key = method.strip().lower()
        if key not in builders:
            raise ValueError(
                f"Unknown balancing technique '{method}'. "
                "Use one or more of: smote, random_under, adasyn, random_over"
            )
        selected[key] = builders[key]
    return selected


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(args.data_path, args.target_column)
    problem_type = infer_problem_type(y)

    print(f"Detected problem type: {problem_type}")
    print(f"Total samples: {len(X)}")
    print(f"Total features after encoding checks: {X.shape[1]}")

    stratify = y if problem_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test = pd.DataFrame(
        imputer.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    distribution_log_rows = []
    results_rows = []

    balancing_methods = [m.strip() for m in args.balancing_techniques.split(",") if m.strip()]
    balancer_builders = get_balancer_builders(balancing_methods)

    if problem_type != "classification":
        print(
            "Regression target detected: class imbalance techniques are skipped "
            "because they are classification-specific."
        )
        balancer_builders = {"none": lambda _rs: None}

    for technique_name, balancer_builder in balancer_builders.items():
        before_dist = class_distribution(y_train)
        print(f"\n[{technique_name}] Class distribution before resampling:")
        print(before_dist.to_string())

        if problem_type == "classification":
            balancer_for_log = balancer_builder(args.random_state)
            _, y_resampled = balancer_for_log.fit_resample(X_train, y_train)
            after_dist = class_distribution(pd.Series(y_resampled))
        else:
            after_dist = before_dist.copy()

        print(f"[{technique_name}] Class distribution after resampling:")
        print(after_dist.to_string())

        for class_label, count in before_dist.items():
            distribution_log_rows.append(
                {
                    "Balancing Technique": technique_name,
                    "Stage": "before",
                    "Class": class_label,
                    "Count": int(count),
                }
            )
        for class_label, count in after_dist.items():
            distribution_log_rows.append(
                {
                    "Balancing Technique": technique_name,
                    "Stage": "after",
                    "Class": class_label,
                    "Count": int(count),
                }
            )

        lasso_resampler = balancer_builder(args.random_state) if problem_type == "classification" else None
        lasso_df = lasso_feature_selection(
            X_train=X_train,
            y_train=y_train,
            problem_type=problem_type,
            resampler=lasso_resampler,
            random_state=args.random_state,
        )
        lasso_top_features = lasso_df["Feature"].head(args.top_n).tolist()

        plot_top_features(
            feature_df=lasso_df,
            feature_col="Feature",
            score_col="LASSO_AbsCoefficient",
            title=f"LASSO Coefficients - {technique_name}",
            output_path=outputs_dir / f"lasso_coefficients_{technique_name}.png",
            top_n=args.top_n,
        )

        results_rows.append(
            {
                "Feature Selection Method": "LASSO",
                "Balancing Technique": technique_name,
                "Important Features": ", ".join(lasso_top_features),
            }
        )

        rf_resampler = balancer_builder(args.random_state) if problem_type == "classification" else None
        rf_df = rf_feature_selection(
            X_train=X_train,
            y_train=y_train,
            problem_type=problem_type,
            resampler=rf_resampler,
            random_state=args.random_state,
        )
        rf_top_features = rf_df["Feature"].head(args.top_n).tolist()

        plot_top_features(
            feature_df=rf_df,
            feature_col="Feature",
            score_col="RF_Importance",
            title=f"Random Forest Feature Importance - {technique_name}",
            output_path=outputs_dir / f"rf_importance_{technique_name}.png",
            top_n=args.top_n,
        )

        results_rows.append(
            {
                "Feature Selection Method": "RandomForest",
                "Balancing Technique": technique_name,
                "Important Features": ", ".join(rf_top_features),
            }
        )

    results_df = pd.DataFrame(results_rows)
    results_path = outputs_dir / "modeling_part1_results.csv"
    results_df.to_csv(results_path, index=False)

    dist_log_df = pd.DataFrame(distribution_log_rows)
    dist_log_path = outputs_dir / "class_distribution_log.csv"
    dist_log_df.to_csv(dist_log_path, index=False)

    print(f"\nSaved results table: {results_path}")
    print(f"Saved class distribution log: {dist_log_path}")
    print(f"Saved feature plots in: {outputs_dir.resolve()}")


if __name__ == "__main__":
    main()
