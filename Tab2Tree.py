# Schema Robustness Evaluation Project (with Real Adult Dataset and Improved Embedding Fix)

import pandas as pd
import numpy as np
import json
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

# 1. Load Real Adult Dataset from UCI
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income']
df = pd.read_csv(url, names=columns, sep=r',\s*', engine='python')

# Clean missing data
df = df.replace('?', np.nan).dropna()

# Drop 'fnlwgt' and 'education-num' for simplicity
df = df.drop(columns=['fnlwgt', 'education-num'])

# 2. Schema Shift: Rename Columns
rename_map = {col: f"col_{col}" for col in df.columns if col != 'income'}
df_renamed = df.rename(columns=rename_map)

# 3. Category Merge Example
df['marital-status'] = df['marital-status'].replace({'Divorced': 'Formerly-Married', 'Widowed': 'Formerly-Married'})

# 4. Feature Composition Example
df['age_bracket'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=['Young','Middle','Senior'])

# 5. Convert Dataset to JSON and XML (sampled)
json_data = []
xml_data = []
for _, row in df.sample(n=100, random_state=42).iterrows():
    rec = row.to_dict()
    json_rec = {
        "demographics": {k: rec[k] for k in ['age','sex','race','marital-status']},
        "education": rec['education'],
        "work": {"workclass": rec['workclass'], "hours-per-week": rec['hours-per-week']},
        "income": rec['income']
    }
    json_data.append(json_rec)

    root = ET.Element("Person")
    dem = ET.SubElement(root, "Demographics")
    for k in ['age','sex','race','marital-status']:
        ET.SubElement(dem, k.replace('-','_')).text = str(rec[k])
    wk = ET.SubElement(root, "Work")
    ET.SubElement(wk, "workclass").text = rec['workclass']
    ET.SubElement(wk, "hours_per_week").text = str(rec['hours-per-week'])
    ET.SubElement(root, "income").text = rec['income']
    xml_data.append(ET.tostring(root).decode())

# 6. Structural Similarity Index (SSI)
def structural_similarity(json1, json2):
    return SequenceMatcher(None, json.dumps(json1, sort_keys=True), json.dumps(json2, sort_keys=True)).ratio()

# 7. Key Alignment Accuracy (KAA)
def key_alignment(json1, json2):
    keys1 = set(json.dumps(json1).replace('{','').replace('}','').split(':'))
    keys2 = set(json.dumps(json2).replace('{','').replace('}','').split(':'))
    if not keys1: return 0
    return len(keys1 & keys2) / len(keys1)

# 8. Prepare Train/Test Data
le = LabelEncoder()
y = le.fit_transform(df['income'])
X = pd.get_dummies(df.drop(columns='income'))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 9. Train Models
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0)
xgb_clf.fit(X_train, y_train)
mlp_clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=100, random_state=0)
mlp_clf.fit(X_train, y_train)

# 10. Evaluate on Original Test
y_pred_xgb = xgb_clf.predict(X_test)
y_pred_mlp = mlp_clf.predict(X_test)
print("Original XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Original MLP Accuracy:", accuracy_score(y_test, y_pred_mlp))

# 11. Simulate Renamed Schema on Test
X_test_renamed = X_test.rename(columns={col: f"col_{col}" for col in X_test.columns})
X_test_renamed_aligned = X_test_renamed.reindex(columns=X_train.columns, fill_value=0)
y_pred_xgb_bad = xgb_clf.predict(X_test_renamed_aligned)
y_pred_mlp_bad = mlp_clf.predict(X_test_renamed_aligned)
print("Renamed (no fix) XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb_bad))
print("Renamed (no fix) MLP Accuracy:", accuracy_score(y_test, y_pred_mlp_bad))

# 12. Dictionary Mapping Fix
inv_map = {f"col_{col}": col for col in X_train.columns}
X_test_mapped = X_test_renamed.rename(columns=inv_map)
X_test_mapped_aligned = X_test_mapped.reindex(columns=X_train.columns, fill_value=0)
y_pred_xgb_corr = xgb_clf.predict(X_test_mapped_aligned)
y_pred_mlp_corr = mlp_clf.predict(X_test_mapped_aligned)
print("Renamed + Mapping XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb_corr))
print("Renamed + Mapping MLP Accuracy:", accuracy_score(y_test, y_pred_mlp_corr))

# 13. Embedding-Based Fix (Improved One-to-One Matching)
def match_columns_by_embedding(renamed_cols, train_cols):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    emb_renamed = model.encode(renamed_cols)
    emb_train = model.encode(train_cols)
    sim_matrix = cosine_similarity(emb_renamed, emb_train)
    mapping = {}
    used = set()
    for i, col in enumerate(renamed_cols):
        sim_matrix[i][list(used)] = -1  # Mask already used matches
        best = sim_matrix[i].argmax()
        mapping[col] = train_cols[best]
        used.add(best)
    return mapping

embedding_map = match_columns_by_embedding(X_test_renamed.columns.tolist(), X_train.columns.tolist())
X_test_emb_mapped = X_test_renamed.rename(columns=embedding_map)
X_test_emb_aligned = X_test_emb_mapped.reindex(columns=X_train.columns, fill_value=0)
y_pred_xgb_emb = xgb_clf.predict(X_test_emb_aligned)
y_pred_mlp_emb = mlp_clf.predict(X_test_emb_aligned)
print("Renamed + Embedding XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb_emb))
print("Renamed + Embedding MLP Accuracy:", accuracy_score(y_test, y_pred_mlp_emb))

# 14. Evaluate Structural Metrics
ssi_scores = []
kaa_scores = []
for i in range(min(50, len(json_data))):
    json_shifted = json.loads(json.dumps(json_data[i]).replace('workclass','employment').replace('marital-status','maritalstatus'))
    ssi_scores.append(structural_similarity(json_data[i], json_shifted))
    kaa_scores.append(key_alignment(json_data[i], json_shifted))

print("Avg Structural Similarity Index (SSI):", np.mean(ssi_scores))
print("Avg Key Alignment Accuracy (KAA):", np.mean(kaa_scores))