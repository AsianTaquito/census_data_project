"""Adult income classification: dissect the data, train the models, save results.

    python main.py           full run, all 7 models
    python main.py --no-nn   kNN + Decision Tree only, a few seconds

Then view the dashboard:

    streamlit run app.py
"""
import warnings
warnings.filterwarnings('ignore')

import argparse
import os

import joblib
import matplotlib
from sklearn.model_selection import train_test_split

SHOW_PLOTS = False   # True re-enables the old blocking matplotlib windows
if not SHOW_PLOTS:
    matplotlib.use('Agg')

import data
import models


def run_pipeline(include_nn=True):
    df = data.load_data()
    data.dissect_data(df)

    data.explore_dataframe(df)
    data.visualization(df, show=SHOW_PLOTS)

    X, y, preprocessor = data.preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    results_df, curves = models.train_and_evaluate(
        X_train, X_test, y_train, y_test, preprocessor,
        include_nn=include_nn, show_plots=SHOW_PLOTS,
    )

    os.makedirs(models.RESULTS_DIR, exist_ok=True)
    results_df.to_csv(models.RESULTS_CSV, index=False)
    joblib.dump(curves, models.CURVES_FILE, compress=3)

    print(f'\nResults saved to {models.RESULTS_CSV}')
    print('View the dashboard with:  streamlit run app.py')
    return results_df


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--no-nn', action='store_true',
                    help='skip the neural networks (a couple of seconds instead of minutes)')
    args = ap.parse_args()
    run_pipeline(include_nn=not args.no_nn)


if __name__ == '__main__':
    main()
