# fraud_models.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# ---------- Helper functions ----------
def load_data(path):
    df = pd.read_csv(path)
    return df

def safe_parse_datetime(series):
    # parse a variety of date strings to datetime, coerce errors
    return pd.to_datetime(series, errors='coerce', infer_datetime_format=True)

def preprocess(df):
    # Make a copy
    df = df.copy()

    # LABEL: prefer explicit column if present
    if 'is_fraud' in df.columns:
        y = df['is_fraud'].astype(float).fillna(0).astype(int)
    else:
        # fallback: detect merchant names prefixed with 'fraud_'
        y = df['merchant'].astype(str).str.startswith('fraud').astype(int)

    # Basic feature engineering
    # 1) Convert unix_time to datetime if present
    if 'unix_time' in df.columns:
        try:
            df['txn_dt'] = pd.to_datetime(df['unix_time'], unit='s', errors='coerce')
        except Exception:
            df['txn_dt'] = safe_parse_datetime(df['unix_time'])
    else:
        df['txn_dt'] = pd.NaT

    # 2) Parse DOB if present and compute age at transaction
    if 'dob' in df.columns:
        dob = safe_parse_datetime(df['dob'])
        # If txn_dt is missing, try using year_date/month_date/day_date if available
        if df['txn_dt'].isna().all():
            # try to build txn_dt from day_date/month_date/year_date
            if {'day_date','month_date','year_date'}.issubset(df.columns):
                # month_date may be names like 'December' -> try parse with day=15 default
                def make_txn(row):
                    try:
                        md = row['month_date']
                        yr = int(row['year_date'])
                        dy = int(row['day_date'])
                        # if month given as name, parse
                        try:
                            m = datetime.strptime(md.strip(), "%B").month
                        except Exception:
                            try:
                                m = int(md)
                            except Exception:
                                m = 1
                        return datetime(yr, m, min(max(1, dy),28))
                    except Exception:
                        return pd.NaT
                df['txn_dt'] = df.apply(make_txn, axis=1)
        age = (df['txn_dt'] - dob).dt.days // 365
        df['age'] = age.fillna(-1).astype(int)
    else:
        df['age'] = -1

    # 3) Extract time features from txn_dt or fallback to hour/minute/seconds columns
    if 'txn_dt' in df.columns and not df['txn_dt'].isna().all():
        df['txn_hour']   = df['txn_dt'].dt.hour.fillna(-1).astype(int)
        df['txn_weekday']= df['txn_dt'].dt.weekday.fillna(-1).astype(int)
        df['txn_month']  = df['txn_dt'].dt.month.fillna(-1).astype(int)
    else:
        for c in ['hour','minute','seconds','day_of_week','month_date']:
            if c in df.columns:
                # coerce to numeric when possible, else -1
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(-1).astype(int)
        # map day_of_week text to number
        if 'day_of_week' in df.columns and df['day_of_week'].dtype == object:
            dow_map = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
            df['txn_weekday'] = df['day_of_week'].map(dow_map).fillna(-1).astype(int)
        else:
            df['txn_weekday'] = df.get('day_of_week', -1).astype(int)
        df['txn_hour'] = df.get('hour', -1).astype(int)
        df['txn_month'] = df.get('month_date', -1).astype(int)

    # 4) Numeric features
    numeric_feats = []
    for c in ['amt', 'merch_lat', 'merch_long', 'lat', 'long', 'city_pop']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            numeric_feats.append(c)

    # 5) Categorical features (low-cardinality): merchant, category, city, state, job
    cat_feats = []
    for c in ['merchant','category','city','state','job']:
        if c in df.columns:
            # trim and fill
            df[c] = df[c].astype(str).str.strip().fillna('missing')
            cat_feats.append(c)

    # 6) Drop obviously PII or identifiers that should not be used raw
    drop_cols = []
    for c in ['trans_num','cc_num','first','last','street','dob','unix_time','txn_dt','day_date','month_date','year_date']:
        if c in df.columns:
            drop_cols.append(c)

    # Final feature list
    engineered_feats = ['age','txn_hour','txn_weekday','txn_month']
    X = df[numeric_feats + cat_feats + engineered_feats].copy()

    return X, y, numeric_feats, cat_feats, engineered_feats

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

    # Model 1: Logistic Regression baseline
    pipe_lr = Pipeline([
        ('pre', preprocessor),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', solver='saga'))
    ])

    # Fit baseline
    print("Training Logistic Regression baseline...")
    pipe_lr.fit(X_train, y_train)
    y_pred_lr = pipe_lr.predict(X_test)
    y_score_lr = pipe_lr.predict_proba(X_test)[:,1]

    # Evaluate
    print("\n--- Logistic Regression performance ---")
    print("ROC AUC:", roc_auc_score(y_test, y_score_lr))
    print("Precision:", precision_score(y_test, y_pred_lr, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred_lr, zero_division=0))
    print("F1:", f1_score(y_test, y_pred_lr, zero_division=0))
    print("Classification report:\n", classification_report(y_test, y_pred_lr, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_lr))

    # Model 2: Random Forest
    pipe_rf = Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=random_state))
    ])

    print("\nTraining Random Forest...")
    pipe_rf.fit(X_train, y_train)
    y_pred_rf = pipe_rf.predict(X_test)
    y_score_rf = pipe_rf.predict_proba(X_test)[:,1]

    print("\n--- Random Forest performance ---")
    print("ROC AUC:", roc_auc_score(y_test, y_score_rf))
    print("Precision:", precision_score(y_test, y_pred_rf, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred_rf, zero_division=0))
    print("F1:", f1_score(y_test, y_pred_rf, zero_division=0))
    print("Classification report:\n", classification_report(y_test, y_pred_rf, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_rf))

    # Save both pipelines
    joblib.dump(pipe_lr, "fraud_pipe_logreg.joblib")
    joblib.dump(pipe_rf, "fraud_pipe_rf.joblib")
    print("\nSaved models: fraud_pipe_logreg.joblib, fraud_pipe_rf.joblib")

    # Optional: feature importance for Random Forest (after preprocessor)
    # We can extract the column names after preprocessing
    try:
        pre = pipe_rf.named_steps['pre']
        # numeric + engineered features
        num_cols = numeric_feats + engineered_feats
        # onehot feature names
        ohe = pre.named_transformers_['cat'].named_steps['onehot']
        # careful: OneHotEncoder sparse=False used above
        ohe_cols = list(ohe.get_feature_names_out(cat_feats)) if hasattr(ohe, 'get_feature_names_out') else []
        all_cols = num_cols + ohe_cols
        rf = pipe_rf.named_steps['clf']
        importances = pd.Series(rf.feature_importances_, index=all_cols).sort_values(ascending=False).head(30)
        print("\nTop feature importances (Random Forest):")
        print(importances)
    except Exception as e:
        print("Could not compute feature importances:", e)


if __name__ == "__main__":
    # change path if needed
    main(csv_path='data.csv')
