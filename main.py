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
from lightgbm import LGBMClassifier
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

    # === Behavioral Features ===
    # Transactions per user in rolling windows cannot be computed without grouping, skip now.
    # But compute merchant/category novelty and spend deviation.

    # Merchant novelty: first time using this merchant?
    if 'merchant' in df.columns:
        df['merchant_count'] = df.groupby('merchant').cumcount()
        numeric_feats.append('merchant_count')

    # Category usage frequency
    if 'category' in df.columns:
        df['category_count'] = df.groupby('category').cumcount()
        numeric_feats.append('category_count')

    # Amount deviation from user median using cc_num
    if 'cc_num' in df.columns:
        df['user_median_amt'] = df.groupby('cc_num')['amt'].transform('median')
        df['amt_dev'] = df['amt'] - df['user_median_amt']
        numeric_feats.append('amt_dev')

        # === Velocity & Geo Movement Features (Safe Version) ===
        # requires sorting
        df = df.sort_values(['cc_num', 'txn_dt'])

        # previous coordinates
        df['prev_lat'] = df.groupby('cc_num')['merch_lat'].shift(1)
        df['prev_lon'] = df.groupby('cc_num')['merch_long'].shift(1)
        df['prev_dt']  = df.groupby('cc_num')['txn_dt'].shift(1)

        # haversine distance
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            return 2 * R * np.arcsin(np.sqrt(a))

        df['dist_km'] = haversine(
            df['prev_lat'], df['prev_lon'],
            df['merch_lat'], df['merch_long']
        )

        # time difference in hours
        df['time_diff_hours'] = (df['txn_dt'] - df['prev_dt']).dt.total_seconds() / 3600
        df['time_diff_hours'] = df['time_diff_hours'].replace(0, np.nan)

        # speed km/h
        df['speed_kmh'] = df['dist_km'] / df['time_diff_hours']
        df['speed_kmh'] = df['speed_kmh'].clip(lower=0, upper=2000)
        df['speed_kmh'] = df['speed_kmh'].astype(float).fillna(0.0)
        numeric_feats.append('speed_kmh')

        # === SAFE In-Place 24h Rolling Windows ===
        df['txn_24h_count'] = 0.0
        df['amt_24h_mean'] = df['amt']

        for cc, sub in df.groupby('cc_num'):
            s = sub.set_index('txn_dt')

            cnt = s['amt'].rolling('24h').count()
            mean_amt = s['amt'].rolling('24h').mean()

            df.loc[sub.index, 'txn_24h_count'] = cnt.values
            df.loc[sub.index, 'amt_24h_mean'] = mean_amt.values

        df['txn_24h_count'] = df['txn_24h_count'].fillna(0)
        df['amt_24h_mean'] = df['amt_24h_mean'].fillna(df['amt'])

        numeric_feats.append('txn_24h_count')
        numeric_feats.append('amt_24h_mean')

        df = df.sort_index()

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

    # Model: LightGBM
    pipe_lgbm = Pipeline([
        ('pre', preprocessor),
        ('clf', LGBMClassifier(boosting_type='gbdt', num_leaves=64, learning_rate=0.05, n_estimators=400, class_weight='balanced'))
    ])

    log.info("\nTraining LightGBM model...")
    pipe_lgbm.fit(X_train, y_train)

    y_score_lgbm = pipe_lgbm.predict_proba(X_test)[:,1]
    from sklearn.metrics import precision_recall_curve
    prec, rec, thresh = precision_recall_curve(y_test, y_score_lgbm)
    beta = 2
    f2 = (1+beta**2) * (prec*rec) / (beta**2 * prec + rec + 1e-9)
    best_idx = f2[:-1].argmax()
    optimal_threshold = thresh[best_idx]
    y_pred_lgbm = (y_score_lgbm > optimal_threshold).astype(int)
    log.info(f"Optimal threshold (F2): {optimal_threshold}")

    log.info("\n--- LightGBM performance ---")
    log.info("ROC AUC: %s", roc_auc_score(y_test, y_score_lgbm))
    log.info("Precision: %s", precision_score(y_test, y_pred_lgbm, zero_division=0))
    log.info("Recall: %s", recall_score(y_test, y_pred_lgbm, zero_division=0))
    log.info("F1: %s", f1_score(y_test, y_pred_lgbm, zero_division=0))
    log.info("Classification report:\n%s", classification_report(y_test, y_pred_lgbm, zero_division=0))
    log.info("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred_lgbm))

    # Save pipeline
    joblib.dump(pipe_lgbm, "fraud_pipe_lgbm.joblib")
    log.info("\nSaved model: fraud_pipe_lgbm.joblib")


if __name__ == "__main__":
    # change path if needed
    main(csv_path='data.csv')
