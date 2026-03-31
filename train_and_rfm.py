\
import pandas as pd, json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
import joblib
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score


BASE = Path(__file__).resolve().parents[1]
customers = pd.read_csv(BASE/'data'/'customers.csv')
transactions = pd.read_csv(BASE/'data'/'transactions.csv')

# Prepare modeling dataset
df = customers.copy()   # your uploaded dataframe

# Gender mapping
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Features & target
X = df[['Age','Gender','AvgBrowsingTime','AvgPageViews','PastPurchases']]
y = df['Purchase']

numeric_cols = X.columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), numeric_cols)
])

# -------------------------------------
# Train-test split
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# Create dir
BASE = Path(".")
models_dir = BASE / "models"
models_dir.mkdir(exist_ok=True)
# -------------------------------------
# 1. Logistic Regression (TUNED)
# -------------------------------------
lr_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(max_iter=500, solver="liblinear"))
])

lr_params = {
    "clf__C": [0.01, 0.1, 1, 5, 10],
    "clf__class_weight": [None, "balanced"]
}

lr_search = RandomizedSearchCV(
    lr_pipe, lr_params, n_iter=6, cv=5, scoring="roc_auc", random_state=42
)
lr_search.fit(X_train, y_train)
clf_lr = lr_search.best_estimator_

joblib.dump(clf_lr, models_dir / "model_logistic.pkl")

# -------------------------------------
# 2. Random Forest (TUNED + CALIBRATED)
# -------------------------------------
rf_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", RandomForestClassifier(random_state=42))
])

rf_params = {
    "clf__n_estimators": [100, 150, 200],
    "clf__max_depth": [5, 8, 10, None],
    "clf__min_samples_split": [2, 5, 8],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__class_weight": [None, "balanced"]
}

rf_search = RandomizedSearchCV(
    rf_pipe, rf_params, n_iter=8, cv=5, scoring="roc_auc", random_state=42
)
rf_search.fit(X_train, y_train)

# calibrate probability
rf_calibrated = CalibratedClassifierCV(rf_search.best_estimator_, cv=5)
rf_calibrated.fit(X_train, y_train)

joblib.dump(rf_calibrated, models_dir / "model_randomforest.pkl")

# -------------------------------------
# 3. XGBoost (TUNED + CALIBRATED)
# -------------------------------------
xgb_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", XGBClassifier(
        eval_metric="logloss",
        random_state=42
    ))
])

xgb_params = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [3, 5, 7],
    "clf__learning_rate": [0.05, 0.1, 0.2],
    "clf__subsample": [0.7, 0.9, 1.0]
}

xgb_search = RandomizedSearchCV(
    xgb_pipe, xgb_params, n_iter=8, cv=5, scoring="roc_auc", random_state=42
)
xgb_search.fit(X_train, y_train)

xgb_calibrated = CalibratedClassifierCV(xgb_search.best_estimator_, cv=5)
xgb_calibrated.fit(X_train, y_train)

joblib.dump(xgb_calibrated, models_dir / "model_xgboost.pkl")

# -------------------------------------
# Evaluate all
# -------------------------------------
results = {}
models = {
    "LogisticRegression": clf_lr,
    "RandomForest": rf_calibrated,
    "XGBoost": xgb_calibrated
}

for name, model in models.items():

    # Predictions
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    report = classification_report(y_test, preds)

    # Store results
    results[name] = {
        "accuracy": float(acc),
        "roc_auc": float(auc)
    }

    # Print properly formatted
    print("\n==============================")
    print(f"{name}")
    print("==============================")
    print(f"Accuracy  : {acc:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print("\nClassification Report:\n")
    print(report)

(models_dir / "train_results.json").write_text(json.dumps(results, indent=2))

print("\nTraining Completed. Models Saved!")

# =====================================================
# FINAL MODEL RETRAINING ON FULL DATA (FOR SCORING)
# =====================================================

print("\nRetraining best model on FULL dataset for scoring...")

# Select best model based on ROC-AUC
best_model_name = max(results, key=lambda x: results[x]["roc_auc"])
print("Best Model:", best_model_name)

best_model = models[best_model_name]

# Retrain best model on full dataset
best_model.fit(X, y)

# Save final model separately
joblib.dump(best_model, models_dir / "final_model.pkl")

# =====================================================
# CUSTOMER SCORING
# =====================================================

# Predict probabilities for ALL customers
all_probs = best_model.predict_proba(X)[:, 1]

scored_customers = df.copy()
scored_customers["Purchase_Probability"] = all_probs
scored_customers["Purchase_Rank"] = scored_customers["Purchase_Probability"].rank(
    ascending=False
)

# Top 20% customers
top_20_threshold = np.percentile(all_probs, 80)
scored_customers["Target_Top20"] = (
    scored_customers["Purchase_Probability"] >= top_20_threshold
).astype(int)

# Save scored dataset
scored_path = models_dir / "scored_customers.csv"
scored_customers.to_csv(scored_path, index=False)

print("Customer scoring completed. Saved to:", scored_path)

# RFM
tx = transactions.copy()
tx['PurchaseDate'] = pd.to_datetime(tx['PurchaseDate'])
snapshot = tx['PurchaseDate'].max() + pd.Timedelta(days=1)
rfm = tx.groupby('CustomerID').agg(Recency=('PurchaseDate', lambda x: (snapshot - x.max()).days),
                                   Frequency=('CustomerID','count'),
                                   Monetary=('TotalAmount','sum')).reset_index()

all_cust = pd.DataFrame({'CustomerID': customers['CustomerID']})
rfm = all_cust.merge(rfm, on='CustomerID', how='left').fillna({'Recency':999,'Frequency':0,'Monetary':0})

rfm['R_score'] = pd.qcut(rfm['Recency'].rank(method='first'), 5, labels=[5,4,3,2,1]).astype(int)
rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
rfm['M_score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
rfm['RFM_Score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']
rfm_path = models_dir/'rfm_customers.csv'
rfm.to_csv(rfm_path, index=False)

# KMeans segmentation
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])
kmeans = KMeans(n_clusters=4, random_state=42).fit(rfm_scaled)
rfm['Cluster'] = kmeans.labels_
rfm.to_csv(rfm_path, index=False)

joblib.dump(kmeans, models_dir/'kmeans_rfm.pkl')
joblib.dump(scaler, models_dir/'rfm_scaler.pkl')
print('RFM & segmentation saved to', rfm_path)
