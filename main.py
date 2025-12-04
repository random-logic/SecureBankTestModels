# fraud_models.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

import logging
# logging.basicConfig(filename='model_output.log', level=logging.INFO, format='%(message)s')
log = logging.getLogger()
log.setLevel(logging.INFO)

file_handler = logging.FileHandler("output.txt")
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

log.addHandler(file_handler)
log.addHandler(console_handler)

# ---------- Helper functions ----------
def load_data(path):
    df = pd.read_csv(path)
    return df

def safe_parse_datetime(series):
    # parse a variety of date strings to datetime, coerce errors
    return pd.to_datetime(series, errors='coerce')

def preprocess(df):
    # Make a copy
    df = df.copy()

    # LABEL: prefer explicit column if present
    if 'is_fraud' in df.columns:
        y = df['is_fraud'].astype(float).fillna(0).astype(int)
    else:
        # fallback: detect merchant names prefixed with 'fraud_'
        y = df['merchant'].astype(str).str.startswith('fraud').astype(int)

    # Compute transaction datetime from 'trans_date_trans_time' or 'unix_time'
    if 'trans_date_trans_time' in df.columns:
        df['txn_dt'] = safe_parse_datetime(df['trans_date_trans_time'])
    elif 'unix_time' in df.columns:
        try:
            df['txn_dt'] = pd.to_datetime(df['unix_time'], unit='s', errors='coerce')
        except Exception:
            df['txn_dt'] = safe_parse_datetime(df['unix_time'])
    else:
        df['txn_dt'] = pd.NaT

    # Numeric features
    numeric_feats = []
    for c in ['amt', 'merch_lat', 'merch_long']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            numeric_feats.append(c)

    # Add engineered time features
    df['txn_hour'] = df['txn_dt'].dt.hour.fillna(-1).astype(int)
    df['txn_weekday'] = df['txn_dt'].dt.weekday.fillna(-1).astype(int)
    df['txn_month'] = df['txn_dt'].dt.month.fillna(-1).astype(int)
    numeric_feats.extend(['txn_hour', 'txn_weekday', 'txn_month'])

    # Add amount log feature
    df['amt_log'] = np.log1p(df['amt'])
    numeric_feats.append('amt_log')

    # Categorical features
    cat_feats = []
    for c in ['merchant','category']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().fillna('missing')
            cat_feats.append(c)

    # Drop all other columns except the ones we keep and the label
    keep_cols = set(numeric_feats + cat_feats + ['txn_dt'])
    drop_cols = [c for c in df.columns if c not in keep_cols and c != 'is_fraud']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Final feature list
    X = df[numeric_feats + cat_feats].copy()

    return X, y, numeric_feats, cat_feats, []

# ---------- Main run ----------
def main(csv_path='securebank.csv', random_state=42):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Put your CSV at this path or change csv_path.")

    df = load_data(path)
    X, y, numeric_feats, cat_feats, engineered_feats = preprocess(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # For categorical: we use OneHot for small cardinality - fallback to handle_unknown
    # For compatibility with sklearn >=1.4 (sparse -> sparse_output)
    try:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
    except TypeError:
        # Fallback for older sklearn versions
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_feats + engineered_feats),
        ('cat', categorical_transformer, cat_feats)
    ], remainder='drop', verbose_feature_names_out=False)

    # Model: Random Forest
    pipe_rf = Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=random_state))
    ])

    log.info("\nTraining Random Forest with Hard Negative Mining...")
    pipe_rf.fit(X_train, y_train)

    for i in range(3):  # three rounds of mining
        y_pred_iter = pipe_rf.predict(X_train)
        y_score_iter = pipe_rf.predict_proba(X_train)[:,1]
        hard_negatives = ( (y_train == 1) & (y_score_iter < 0.4) )  # harder fraud cases near decision boundary
        if hard_negatives.sum() == 0:
            break
        X_train_iter = pd.concat([X_train, X_train[hard_negatives]], ignore_index=True)
        y_train_iter = pd.concat([y_train, y_train[hard_negatives]], ignore_index=True)
        pipe_rf.fit(X_train_iter, y_train_iter)
        log.info(f"Iteration {i+1}: Re-trained with {hard_negatives.sum()} hard negatives")

    y_score_rf = pipe_rf.predict_proba(X_test)[:,1]
    threshold = 0.3
    y_pred_rf = (y_score_rf > threshold).astype(int)
    log.info(f"Applied probability threshold: {threshold}")

    log.info("\n--- Random Forest performance ---")
    log.info("ROC AUC: %s", roc_auc_score(y_test, y_score_rf))
    log.info("Precision: %s", precision_score(y_test, y_pred_rf, zero_division=0))
    log.info("Recall: %s", recall_score(y_test, y_pred_rf, zero_division=0))
    log.info("F1: %s", f1_score(y_test, y_pred_rf, zero_division=0))
    log.info("Classification report:\n%s", classification_report(y_test, y_pred_rf, zero_division=0))
    log.info("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred_rf))

    # Save pipeline
    joblib.dump(pipe_rf, "fraud_pipe_rf.joblib")
    log.info("\nSaved model: fraud_pipe_rf.joblib")


if __name__ == "__main__":
    # change path if needed
    main(csv_path='data.csv')
