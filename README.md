# csv
airline dataset passenger satisfaction

## Modeling Part 1 Workflow

This repository now includes a complete, reproducible Modeling Part 1 workflow for:
- train/test split,
- class imbalance handling,
- feature selection,
- and report-ready output generation.

### File added
- `/home/runner/work/csv/csv/modeling_part1.py`

### Requirements
Install required libraries:

```bash
pip install pandas scikit-learn imbalanced-learn matplotlib seaborn
```

### Example run

```bash
python /home/runner/work/csv/csv/modeling_part1.py \
  --data-path "/home/runner/work/csv/csv/Airline Passenger Satisfaction.csv" \
  --target-column satisfaction \
  --test-size 0.2 \
  --random-state 42 \
  --top-n 10 \
  --balancing-techniques smote,random_under \
  --outputs-dir "/home/runner/work/csv/csv/outputs"
```

### Outputs
Generated under `outputs/`:
- `modeling_part1_results.csv` (required results table)
- `class_distribution_log.csv` (before/after class counts by technique)
- `lasso_coefficients_<technique>.png`
- `rf_importance_<technique>.png`

### Supporting docs
- `/home/runner/work/csv/csv/REPORT.md`
- `/home/runner/work/csv/csv/SLIDES_OUTLINE.md`
