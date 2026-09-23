# Adult Income Classification (UCI Census)

This project predicts whether an individual's income exceeds $50K/year using the UCI Adult (Census) dataset. It provides an end-to-end workflow: data exploration, preprocessing, model training, evaluation, and visual comparisons across classical ML models and Neural Networks.

Results are explored in a **Streamlit dashboard** (`app.py`): stat tiles, the
dataset dissection, the model comparison, and a sortable results table.

---

## Overview
- Binary classification of income category: `<=50K` vs `>50K`.
- Demographic features such as age, education, occupation, hours-per-week, etc.
- Reproducible pipeline: dissection → preprocessing → training → evaluation → dashboard.
- `main.py` trains and saves; `app.py` reads those results and draws them.

Only 24.1% of rows are above $50K, so **accuracy alone is misleading** - always
guessing `<=50K` scores 75.9%. The expanded metrics exist to make that visible.

---

## Dataset
- Source: Kaggle — [`uciml/adult-census-income`](https://www.kaggle.com/datasets/uciml/adult-census-income).
- Fetched via `kagglehub.dataset_download('uciml/adult-census-income')` and cached locally.
- Instances: 32,561 | Attributes: 14 + target (`income`, renamed to `class`).

`load_data()` normalizes this copy on the way in: dotted column names become
dashed (`hours.per.week` → `hours-per-week`), `income` becomes `class`, and the
`?` missing-value marker is read as `NaN` so the imputers actually see it
(1,836 in `workclass`, 1,843 in `occupation`, 583 in `native-country`).

Main features include: `age`, `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `hours-per-week`, `native-country`.

---

## Models
Defined in `build_models()` in `models.py`, the pipeline compares:
- k-Nearest Neighbors: `KNeighborsClassifier(n_neighbors=5)`.
- Decision Tree: `DecisionTreeClassifier(max_depth=8, random_state=42)`.
- Neural Networks (MLPClassifier), when enabled:
  - NN: 1 layer (50).
  - NN: 2 layers (100, 50).
  - NN: 3 layers (250, 100, 50).
  - NN: Slow Learning Rate (`learning_rate_init=1e-6`).
  - NN: Extensive Training (`max_iter=5000`).

Each model reports Accuracy, Balanced Accuracy, Precision, Recall, Specificity,
F1 (>50K), F1-macro, ROC-AUC, PR-AUC, MCC, the four confusion-matrix counts
(TN/FP/FN/TP), and both training and prediction time.

---

## Dataset dissection
Before any model runs, `dissect_data()` reports the share of each group earning
over $50K - by education, occupation, marital status, age band, hours-per-week
band and sex - dropping any group under 50 rows. Groups are printed to the
console and shown on the dashboard's **The data** tab.

---

## Project structure
| File | Holds |
|---|---|
| `data.py` | loading, cleaning, exploration, dissection, preprocessing |
| `models.py` | model definitions, training, metrics |
| `app.py` | the Streamlit dashboard |
| `main.py` | argument parsing and orchestration only |
| `model_results/` | saved metrics and scores from the last run |

---

## Quick Start (Windows)

1) Clone the repository
```bash
git clone https://github.com/AsianTaquito/census_data_miniProject.git
cd census_data_miniProject
```

2) Install dependencies
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install pandas numpy matplotlib seaborn scikit-learn joblib kagglehub streamlit
```

3) Train the models
```bash
python main.py           # all 7 models (~9 min)
python main.py --no-nn   # kNN + Decision Tree only, a few seconds
```
This writes `model_results/model_results.csv` and `model_results/model_curves.pkl`.

4) Open the dashboard
```bash
streamlit run app.py
```
It reads whatever `main.py` last saved, so you can restyle the charts and hit
**R** to rerun without retraining. The census data and results are cached with
`@st.cache_data`; use **C** to clear the cache if you re-run the pipeline while
the app is open.

---

## Configuration
- Neural Networks are included by default; `--no-nn` skips them.
- `SHOW_PLOTS = True` at the top of `main.py` re-enables the old blocking
  matplotlib windows (one per chart, plus one confusion matrix per model). They
  are off by default because the dashboard already covers them.
- Adjust hyperparameters (e.g., depths, hidden layers, learning rate, `max_iter`) in `build_models()` in `models.py`.
- Chart colors live in `SERIES` / `RAMP` at the top of `app.py`, and follow
  Streamlit's light/dark theme. Both palettes are validated for colorblind
  separation and contrast against their own surface.

---

## Outputs
- `model_results/model_results.csv`: all 16 metrics per model.
- `model_results/model_curves.pkl`: held-out labels and per-model scores, so the
  dashboard can redraw ROC curves without refitting.


---

## Notes
- Requires internet access on first run; `kagglehub` caches the download under `~/.cache/kagglehub`.
- Neural Networks can be slower to train; reduce `max_iter` or disable via `include_nn=False` for quicker runs.
- Warnings are suppressed at runtime for cleaner logs.
