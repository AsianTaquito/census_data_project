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
from sklearn.neural_network import MLPClassifier


# Load Dataset
def load_data():
    print("...Loading dataset from OpenML...")
    df = fetch_openml('adult', version=2, as_frame=True).frame
    print("Dataset loaded successfully.")
    return df


# Dataset Exploration
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
def visualization(df):
    print("\n...Generating data visualizations...")

    # Age distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['age'], bins=30, kde=True)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.tight_layout
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


# Model Training 
def timed_fit(pipeline, X_train, y_train):
    start = time.time()
    pipeline.fit(X_train, y_train)
    return time.time() - start


def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, include_nn=True):
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

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-macro': f1_macro,
            'Train Time (s)': round(train_time, 2)
        })

    #comparison plot
    results_df = pd.DataFrame(results)
    print('\n--- MODEL PERFORMANCE ---')
    print(results_df.sort_values(by='F1-macro', ascending=False))

    #drop train time to make seperate plot
    metrics_only = results_df.drop(columns=['Train Time (s)'])

    melted = metrics_only.melt(id_vars='Model', var_name='Metric', value_name='Score')
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x='Metric', y='Score', hue='Model', palette='coolwarm')
    plt.title('Model Comparison on Adult Income Dataset')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=results_df, x='Model', y='Train Time (s)', palette='coolwarm')
    plt.title('Training Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return results_df


# Main
def main():
    df = load_data()
    explore_dataframe(df)
    visualization(df)

    X, y, preprocessor = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    results_df = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)
    results_df.to_csv('model_results.csv', index=False)
    print('\nResults saved to model_results.csv')


if __name__ == '__main__':
    main()