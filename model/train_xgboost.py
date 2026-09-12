"""
Trains the fast-path XGBoost classifier on lexical/structural URL features.
Binary target: 0 = benign, 1 = malicious (phishing/malware/defacement collapsed).
Also keeps the original multiclass label so the extension can show threat type.
"""
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib

from features import extract_features, FEATURE_NAMES

DATA_PATH = '../data/malicious_phish.csv'
MODEL_OUT = 'xgb_model.json'
LABEL_ENCODER_OUT = 'label_encoder.joblib'

print('Loading dataset...')
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['url', 'type'])
print(f'{len(df)} rows loaded')

# The base dataset's benign URLs are almost all bare domain/path listings
# with no query strings, which trains the model to over-flag any URL with
# params as suspicious. Oversample a small hand-built set of realistic
# benign query-string URLs (search, video, e-commerce, docs, etc.) to
# correct that gap -- see generate_benign_augmentation.py for the source.
try:
    aug = pd.read_csv('../data/synthetic_benign.csv')
    OVERSAMPLE_FACTOR = 60
    aug_oversampled = pd.concat([aug] * OVERSAMPLE_FACTOR, ignore_index=True)
    df = pd.concat([df, aug_oversampled], ignore_index=True)
    print(f'Added {len(aug_oversampled)} oversampled benign-with-params rows '
          f'({len(aug)} unique x{OVERSAMPLE_FACTOR}). New total: {len(df)}')
except FileNotFoundError:
    print('No synthetic_benign.csv found, skipping augmentation')

# Binary target for blocking decision
df['is_malicious'] = (df['type'] != 'benign').astype(int)

# Multiclass label for the "why flagged" reason shown in the popup
le = LabelEncoder()
df['type_encoded'] = le.fit_transform(df['type'])
joblib.dump(le, LABEL_ENCODER_OUT)
print('Classes:', list(le.classes_))

print('Extracting features (this takes a few minutes on 650k rows)...')
t0 = time.time()
feat_rows = []
for i, u in enumerate(df['url'].values):
    feat_rows.append(extract_features(str(u)))
    if i % 100000 == 0:
        print(f'  {i}/{len(df)}  ({time.time()-t0:.0f}s)')

X = pd.DataFrame(feat_rows, columns=FEATURE_NAMES)
y_bin = df['is_malicious'].values
y_multi = df['type_encoded'].values
print(f'Feature extraction done in {time.time()-t0:.0f}s')

X_train, X_test, ybin_train, ybin_test, ymulti_train, ymulti_test = train_test_split(
    X, y_bin, y_multi, test_size=0.15, random_state=42, stratify=y_bin
)

print('Training binary (malicious vs benign) model...')
bin_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric='logloss',
    tree_method='hist',
    n_jobs=-1,
    random_state=42,
)
bin_model.fit(X_train, ybin_train)

pred = bin_model.predict(X_test)
proba = bin_model.predict_proba(X_test)[:, 1]
print('\n--- Binary classification report ---')
print(classification_report(ybin_test, pred, target_names=['benign', 'malicious']))
print('ROC-AUC:', roc_auc_score(ybin_test, proba))
print('Confusion matrix:\n', confusion_matrix(ybin_test, pred))

print('\nTraining multiclass (threat-type) model...')
multi_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric='mlogloss',
    tree_method='hist',
    n_jobs=-1,
    random_state=42,
)
multi_model.fit(X_train, ymulti_train)
pred_multi = multi_model.predict(X_test)
print('\n--- Multiclass classification report ---')
print(classification_report(ymulti_test, pred_multi, target_names=le.classes_))

bin_model.save_model('xgb_binary.json')
multi_model.save_model('xgb_multiclass.json')
print('\nSaved xgb_binary.json, xgb_multiclass.json, label_encoder.joblib')

# feature importance snapshot for the report/paper
importances = pd.Series(bin_model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
print('\nTop 10 features (binary model):')
print(importances.head(10))
importances.to_csv('feature_importance.csv')
