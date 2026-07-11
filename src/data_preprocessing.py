"""
Data Exploration and Preprocessing
=====================
Data cleaning, feature engineering, and preprocessing pipeline for the 
Amazon Sales Dataset used in ensemble classifier fusion analysis.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


def load_data(filepath: str) -> pd.DataFrame:
    """Load the Amazon sales dataset from CSV."""
    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Drop irrelevant columns (IDs, URLs)."""
    drop_cols = [c for c in df.columns if 'id' in c.lower() or 'url' in c.lower()]
    df = df.drop(columns=drop_cols)
    print(f"Dropped columns: {drop_cols}")
    return df


def parse_currency_and_percentage(df: pd.DataFrame) -> pd.DataFrame:
    """Convert price and percentage columns from strings to floats."""
    price_cols = ['discounted_price', 'actual_price']
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
    
    if 'discount_percentage' in df.columns:
        df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%', '').astype(float)
    
    if 'rating_count' in df.columns:
        df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '').astype(float)
    
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using median (numeric) and mode (categorical)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Numeric imputation
    num_imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    
    # Categorical imputation
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    return df


def filter_rare_classes(df: pd.DataFrame, target_col: str, min_count: int = 6) -> pd.DataFrame:
    """Remove classes with fewer than min_count samples."""
    class_counts = df[target_col].value_counts()
    valid_classes = class_counts[class_counts >= min_count].index
    df_filtered = df[df[target_col].isin(valid_classes)].copy()
    print(f"Filtered from {class_counts.nunique()} to {len(valid_classes)} classes")
    return df_filtered


def encode_features(df: pd.DataFrame, target_col: str):
    """One-hot encode categorical features and label encode the target."""
    # Separate features and target
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])
    
    # One-hot encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    
    # Label encode target
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    print(f"Feature dimensionality: {X.shape[1]} → {X_encoded.shape[1]}")
    print(f"Number of classes: {len(le.classes_)}")
    
    return X_encoded, y, le


def split_and_scale(X, y, test_size=0.3, random_state=42):
    """Split data and apply standard scaling."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def full_preprocessing_pipeline(filepath: str, target_col: str = 'category'):
    """Execute the complete preprocessing pipeline."""
    print("=" * 60)
    print("STARTING PREPROCESSING PIPELINE")
    print("=" * 60)
    
    df = load_data(filepath)
    df = clean_column_names(df)
    df = parse_currency_and_percentage(df)
    df = impute_missing_values(df)
    df = filter_rare_classes(df, target_col)
    X, y, le = encode_features(df, target_col)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print("=" * 60)
    
    return X_train, X_test, y_train, y_test, scaler, le


if __name__ == "__main__":
    # Example usage
    X_train, X_test, y_train, y_test, scaler, le = full_preprocessing_pipeline(
        filepath="data/amazon.csv"
    )
