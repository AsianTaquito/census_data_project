# Adult Income Classification (UCI Census)

This project predicts whether an individual's income exceeds $50K/year using the UCI Adult (Census) dataset. It provides an end-to-end workflow: data exploration, preprocessing, model training, evaluation, and visual comparisons across classical ML models and Neural Networks.

---

## Overview
- Binary classification of income category: `<=50K` vs `>50K`.
- Demographic features such as age, education, occupation, hours-per-week, etc.
- Reproducible pipeline: preprocessing → training → evaluation → reporting.
- Results saved to `model_results.csv` with plotted confusion matrices and comparison charts.

---

## Dataset
- Source: UCI Machine Learning Repository — Adult Dataset.
- Fetched via OpenML: `fetch_openml('adult', version=2, as_frame=True)`.
- Instances: ~32,000 | Attributes: 14 + target (`class`).

Main features include: `age`, `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `hours-per-week`, `native-country`.

---

## Models
Defined in `main.py`, the pipeline compares:
- k-Nearest Neighbors: `KNeighborsClassifier(n_neighbors=5)`.
- Decision Tree: `DecisionTreeClassifier(max_depth=8, random_state=42)`.
- Neural Networks (MLPClassifier), when enabled:
  - NN: 1 layer (50).
  - NN: 2 layers (100, 50).
  - NN: 3 layers (250, 100, 50).
  - NN: Slow Learning Rate (`learning_rate_init=1e-6`).
  - NN: Extensive Training (`max_iter=5000`).

Each model reports Accuracy, Precision, Recall, F1-macro, and training time, with a confusion matrix. Comparative bar charts visualize metrics and training times.

---

## Quick Start (Windows)

1) Clone the repository
```bash
git clone https://github.com/AsianTaquito/census_data_miniProject.git
cd census_data_miniProject
```

2) Create a virtual environment and install dependencies
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

3) Run the project
```bash
python main.py
```
This downloads the dataset, explores it, trains all configured models, displays plots, and writes results to `model_results.csv`.

---

## Configuration
- Neural Networks are included by default: `train_and_evaluate(..., include_nn=True)` in `main.py`.
- To skip NN training, set `include_nn=False`.
- Adjust hyperparameters (e.g., depths, hidden layers, learning rate, `max_iter`) directly in the `models` dict in `main.py`.

---

## Outputs
- `model_results.csv`: per-model metrics (Accuracy, Precision, Recall, F1-macro) and training time.
- Confusion matrix plots per model.
- Comparative bar charts for metrics and training time.

---

## Notes
- Requires internet access on first run to fetch the dataset from OpenML.
- Neural Networks can be slower to train; reduce `max_iter` or disable via `include_nn=False` for quicker runs.
- Warnings are suppressed at runtime for cleaner logs.
