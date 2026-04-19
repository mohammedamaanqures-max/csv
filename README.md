# csv
airline dataset passenger satisfaction

## Modeling Part 1 Workflow

This repository now includes a complete, reproducible Modeling Part 1 workflow for:
- train/test split,
- class imbalance handling,
- feature selection,
- and report-ready output generation.

### File added
- `modeling_part1.py`

### Requirements
Install required libraries:

```bash
pip install pandas scikit-learn imbalanced-learn matplotlib seaborn
```

### Example run

```bash
python modeling_part1.py \
  --data-path "Airline Passenger Satisfaction.csv" \
  --target-column satisfaction \
  --test-size 0.2 \
  --random-state 42 \
  --top-n 10 \
  --balancing-techniques smote,random_under \
  --outputs-dir outputs
```

### Outputs
Generated under `outputs/`:
- `modeling_part1_results.csv` (required results table)
- `class_distribution_log.csv` (before/after class counts by technique)
- `lasso_coefficients_<technique>.png`
- `rf_importance_<technique>.png`

### Supporting docs
- `REPORT.md`
- `SLIDES_OUTLINE.md`
