"""Loading, cleaning, exploration and dissection of the census data."""
import warnings
warnings.filterwarnings('ignore')

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

KAGGLE_DATASET = 'uciml/adult-census-income'

BANDS = {
    'age': ([16, 25, 35, 45, 55, 65, 100],
            ['17-25', '26-35', '36-45', '46-55', '56-65', '66+']),
    'hours-per-week': ([0, 20, 34, 40, 50, 60, 100],
                       ['1-20', '21-34', '35-40', '41-50', '51-60', '61+']),
}

DISSECT_CUTS = {
    'Education': 'education',
    'Occupation': 'occupation',
    'Marital status': 'marital-status',
    'Age': 'age band',
    'Hours per week': 'hours-per-week band',
    'Sex': 'sex',
}


# Load Dataset
def load_data():
    import kagglehub

    print("...Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    # this copy marks missing as '?' and names columns with dots
    df = pd.read_csv(os.path.join(path, 'adult.csv'),
                     na_values='?', skipinitialspace=True)
    df.columns = [c.replace('.', '-') for c in df.columns]
    df = df.rename(columns={'income': 'class'})
    print(f"Loaded {len(df):,} rows from {path}")
    return df


# Dataset Exploration
def explore_dataframe(df):

    print("\n--- Data Info ---")
    print(df.info())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Income Class Distribution ---")
    print(df['class'].value_counts(normalize=True))


# Dataset dissection
def high_income(df):
    """Boolean series: does this row earn more than 50K."""
    return df['class'].astype(str).str.strip().str.startswith('>50')


def rate_table(df, col, min_n=50, by_rate=True):
    """Share of >50K earners per level of col, with the group size.

    Groups under min_n rows are dropped. Banded columns keep their natural
    order (by_rate=False) so the trend stays readable.
    """
    grouped = df.groupby(col, observed=True)
    out = pd.DataFrame({'n': grouped.size(), 'rate': grouped['_high'].mean()})
    out = out[out['n'] >= min_n]
    return out.sort_values('rate') if by_rate else out


def dissect_data(df, verbose=True):
    """Break the census rows down by the features that actually move income."""
    df = df.copy()
    df['_high'] = high_income(df)
    for col, (bins, labels) in BANDS.items():
        df[f'{col} band'] = pd.cut(df[col], bins=bins, labels=labels)

    base_rate = df['_high'].mean()
    banded = {f'{c} band' for c in BANDS}
    tables = {title: rate_table(df, col, by_rate=col not in banded)
              for title, col in DISSECT_CUTS.items()}

    has_gain = df['capital-gain'] > 0
    with_gain = df.loc[has_gain, '_high'].mean()
    without_gain = df.loc[~has_gain, '_high'].mean()

    if verbose:
        print(f"\n--- WHO EARNS >50K (base rate {base_rate:.1%}, n={len(df):,}) ---")
        for title, tbl in tables.items():
            top, bottom = tbl.iloc[-1], tbl.iloc[0]
            print(f"\n{title}: {top.name} {top['rate']:.1%} (n={int(top['n']):,})"
                  f"  vs  {bottom.name} {bottom['rate']:.1%} (n={int(bottom['n']):,})")
            print(tbl.assign(rate=lambda d: (d['rate'] * 100).round(1)).to_string())
        print(f"\nCapital gain > 0: {has_gain.mean():.1%} of rows, {with_gain:.1%} earn "
              f">50K vs {without_gain:.1%} without - a {with_gain - without_gain:+.1%} swing.")

    return {'base_rate': base_rate, 'n_rows': len(df), 'tables': tables,
            'gain_lift': with_gain - without_gain, 'gain_share': has_gain.mean()}


#Data cleaning & preprocessing
def preprocess_data(df):
    df = df.copy()
    df['class'] = df['class'].astype(str).str.strip()

    # Target variable- if income >50K
    y = df['class'].apply(lambda x: 1 if x.startswith('>50') else 0)
    X = df.drop('class', axis=1)

    print("\n--- BEFORE PREPROCESSING ---")
    print(X.dtypes)
    print("\nPreview of data:")
    print(X.head())

    #category detection
    categorical_cols = []
    numeric_cols = []

    for col in X.columns:
        # If pandas detects as object or string dtype
        if X[col].dtype == 'object':
            categorical_cols.append(col)
        # If column has non-numeric entries even though dtype says numeric
        elif X[col].apply(lambda v: isinstance(v, str)).any():
            categorical_cols.append(col)
        # If column has small number of unique values (likely categorical)
        elif X[col].nunique() < 20:
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    print("\n--- COLUMN TYPE SUMMARY ---")
    print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

    # Define transformations
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    #Fit and show post-processing summary
    X_processed = preprocessor.fit_transform(X)

    print("\n--- AFTER PREPROCESSING ---")
    print(f"Transformed feature matrix shape: {X_processed.shape}")

    cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
    all_features = numeric_cols + cat_features.tolist()
    print(f"Total processed features: {len(all_features)}")
    print("Sample feature names:", all_features[:15])

    return X, y, preprocessor


# Visualization
def visualization(df, show=False):
    if not show:
        return
    print("\n...Generating data visualizations...")

    # Age distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['age'], bins=30, kde=True)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.countplot(x="class", data=df, palette="pastel")
    plt.title("Income Class Distribution", fontsize=14)
    plt.xlabel("Income Category")
    plt.tight_layout()
    plt.show()

    # Education Level Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(
        y="education",
        data=df,
        order=df["education"].value_counts().index,
        palette="muted"
    )
    plt.title("Education Level Distribution", fontsize=14)
    plt.ylabel("Education Level")
    plt.tight_layout()
    plt.show()

    # Gender Distribution
    plt.figure(figsize=(5, 5))
    df["sex"].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#99ff99'])
    plt.title("Gender Distribution", fontsize=14)
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

    # Correlation Heatmap
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    if not numeric_cols.empty:
        plt.figure(figsize=(8, 6))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap="coolwarm", square=True)
        plt.title("Correlation Heatmap (Numeric Features)", fontsize=14)
        plt.tight_layout()
        plt.show()
    else:
        print("No numeric columns found for correlation heatmap.")
