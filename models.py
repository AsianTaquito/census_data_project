"""Model definitions, training and evaluation."""
import warnings
warnings.filterwarnings('ignore')

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

RESULTS_DIR = 'model_results'
RESULTS_CSV = os.path.join(RESULTS_DIR, 'model_results.csv')
CURVES_FILE = os.path.join(RESULTS_DIR, 'model_curves.pkl')

METRIC_COLS = ['Accuracy', 'Balanced Acc', 'Precision', 'Recall', 'Specificity',
               'F1 (>50K)', 'F1-macro', 'ROC-AUC', 'PR-AUC', 'MCC']


def build_models(include_nn=True):
    models = {
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=8),
    }

    # neural network
    if include_nn:
        models.update({
            'NN: 1 Layer (50)': MLPClassifier(hidden_layer_sizes=(50), max_iter=500, random_state=42),
            'NN: 2 layers (100, 50)': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            'NN: 3 layers (250,100,50)': MLPClassifier(hidden_layer_sizes=(250, 100, 50), max_iter=500, random_state=42),
            'NN: Slow Learning Rate': MLPClassifier(hidden_layer_sizes=(100, 50), learning_rate_init=0.000001, max_iter=500, random_state=42),
            'NN: Extensive Training': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=5000, random_state=42)
        })
    return models


# Model Training
def timed_fit(pipeline, X_train, y_train):
    start = time.time()
    pipeline.fit(X_train, y_train)
    return time.time() - start


def score_model(name, y_test, y_pred, y_score, train_time, predict_time):
    """All metrics for one model."""
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Balanced Acc': balanced_accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'Specificity': tn / (tn + fp) if tn + fp else 0.0,
        'F1 (>50K)': f1_score(y_test, y_pred, zero_division=0),
        'F1-macro': f1_score(y_test, y_pred, average='macro'),
        'ROC-AUC': roc_auc_score(y_test, y_score),
        'PR-AUC': average_precision_score(y_test, y_score),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
        'Train Time (s)': round(train_time, 2),
        'Predict Time (s)': round(predict_time, 2),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor,
                       include_nn=True, show_plots=False):
    results = []
    scores = {}

    for name, model in build_models(include_nn).items():
        print(f"\nTraining {name}...")
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        train_time = timed_fit(pipe, X_train, y_train)

        start = time.time()
        y_pred = pipe.predict(X_test)
        predict_time = time.time() - start
        scores[name] = pipe.predict_proba(X_test)[:, 1]

        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K'], zero_division=0))

        if show_plots:
            ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test,
                                                  display_labels=['<=50K', '>50K'],
                                                  cmap='Blues')
            plt.title(f'{name} - Confusion Matrix')
            plt.show()

        results.append(score_model(name, y_test, y_pred, scores[name],
                                   train_time, predict_time))

    results_df = pd.DataFrame(results)
    print('\n--- MODEL PERFORMANCE ---')
    print(results_df.sort_values(by='F1-macro', ascending=False).to_string(index=False))

    return results_df, {'y_test': np.asarray(y_test), 'scores': scores}
