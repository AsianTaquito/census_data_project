import warnings
warnings.filterwarnings('ignore')

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC


# Load Dataset
def load_data():
    print("...Loading dataset from OpenML...")
    df = fetch_openml('adult', version=2, as_frame=True).frame
    print("✅ Dataset loaded successfully.")
    return df


# Dataset Info
def explore_dataframe(df):
    print("\n--- Dataset Overview ---")
    print("Shape:", df.shape)
    print("\nFirst 5 rows:\n", df.head())

    print("\n--- Data Info ---")
    print(df.info())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Income Class Distribution ---")
    print(df['class'].value_counts(normalize=True))


# Visualizations
def plot_eda(df):
    sns.set(style='whitegrid')

    # Age distribution
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df['age'].dropna(), bins=30, kde=True, color='skyblue')
    plt.title('Age Distribution')

    # Income distribution
    plt.subplot(1, 2, 2)
    sns.countplot(x='class', data=df, color='lightcoral')
    plt.title('Income Distribution')
    plt.tight_layout()
    plt.show()

    # Education count
    plt.figure(figsize=(10, 6))
    sns.countplot(y='education', data=df,
                  order=df['education'].value_counts().index,
                  palette='viridis')
    plt.title('Education Levels')
    plt.tight_layout()
    plt.show()

    # Gender pie chart
    plt.figure(figsize=(6, 5))
    df['sex'].value_counts().plot.pie(autopct='%1.1f%%',
                                      startangle=90,
                                      colors=['#66b3ff', '#99ff99'])
    plt.ylabel('')
    plt.title('Gender Distribution')
    plt.tight_layout()
    plt.show()

    # Correlation (numeric features)
    numeric_cols = df.select_dtypes(exclude=['object']).columns
    corr = df[numeric_cols].apply(lambda x: (x - x.mean()) / x.std()).corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation (numeric features)')
    plt.tight_layout()
    plt.show()


# Preprocessing
def preprocess_data(df):
    df = df.copy()
    df['class'] = df['class'].astype(str).str.strip()
    y = df['class'].apply(lambda x: 1 if x.startswith('>50') else 0)

    X = df.drop('class', axis=1)

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    print('\n--- Preprocessing Summary ---')
    print('Numeric columns:', numeric_cols)
    print('Categorical columns:', categorical_cols)

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

    return X, y, preprocessor


# Model Training & Evaluation
def timed_fit(pipeline, X_train, y_train):
    start = time.time()
    pipeline.fit(X_train, y_train)
    return time.time() - start


def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    models = {
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=8),
        'LinearSVC': LinearSVC(max_iter=5000, random_state=42, class_weight='balanced')
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        train_time = timed_fit(pipe, X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro')

        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K'], zero_division=0))

        # Confusion Matrix
        ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test,
                                              display_labels=['<=50K', '>50K'],
                                              cmap='Blues')
        plt.title(f'{name} - Confusion Matrix')
        plt.show()

        # Save model
        joblib.dump(pipe, f"{name.lower().replace(' ', '_')}_model.pkl")

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-macro': f1_macro,
            'Train Time (s)': round(train_time, 2)
        })

    results_df = pd.DataFrame(results)
    print('\n--- MODEL PERFORMANCE ---')
    print(results_df.sort_values(by='F1-macro', ascending=False))

    # Comparison plot
    melted = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    plt.figure(figsize=(10, 5))
    sns.barplot(data=melted, x='Model', y='Score', hue='Metric', palette='coolwarm')
    plt.title('Model Comparison on Adult Income Dataset')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return results_df


# Main
def main():
    df = load_data()
    explore_dataframe(df)
    plot_eda(df)

    X, y, preprocessor = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results_df = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)
    results_df.to_csv('model_results.csv', index=False)
    print('\nResults saved to model_results.csv')


if __name__ == '__main__':
    main()
