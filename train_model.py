"""
OptiCrop AI - Model Training Script
Trains multiple ML models, evaluates them, and saves the best one.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, 'dataset', 'Crop_recommendation.csv')
MODEL_DIR  = os.path.join(BASE_DIR, 'model')
STATIC_DIR = os.path.join(BASE_DIR, 'static', 'images')

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# ── Load Data ──────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(df.head())

# ── EDA ────────────────────────────────────────────────────────────────────────
print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Basic Stats ---")
print(df.describe())

features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
target   = 'label'

# Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(df[features].corr(), annot=True, cmap='YlGn', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, 'heatmap.png'), dpi=100)
plt.close()

# Crop distribution
plt.figure(figsize=(14, 5))
df[target].value_counts().plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Crop Distribution in Dataset', fontsize=14, fontweight='bold')
plt.xlabel('Crop')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, 'crop_distribution.png'), dpi=100)
plt.close()

# Feature boxplots
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, feat in enumerate(features):
    axes[i].boxplot(df[feat], patch_artist=True,
                    boxprops=dict(facecolor='lightgreen', color='darkgreen'))
    axes[i].set_title(feat, fontweight='bold')
    axes[i].set_xlabel('')
axes[-1].set_visible(False)
plt.suptitle('Feature Boxplots', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, 'boxplots.png'), dpi=100)
plt.close()

# Feature histograms
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, feat in enumerate(features):
    axes[i].hist(df[feat], bins=30, color='steelblue', edgecolor='black', alpha=0.8)
    axes[i].set_title(feat, fontweight='bold')
axes[-1].set_visible(False)
plt.suptitle('Feature Histograms', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, 'histograms.png'), dpi=100)
plt.close()

print("EDA plots saved.")

# ── Preprocessing ──────────────────────────────────────────────────────────────
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df[target])

X = df[features].values
y = df['label_enc'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ── Model Training & Evaluation ────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=5),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Naive Bayes':         GaussianNB(),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'model':     model,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall':    recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1':        f1_score(y_test, y_pred, average='weighted', zero_division=0),
    }
    print(f"{name:22s} | Acc: {results[name]['accuracy']:.4f} | "
          f"F1: {results[name]['f1']:.4f}")

# ── Model Comparison Chart ─────────────────────────────────────────────────────
names  = list(results.keys())
accs   = [results[n]['accuracy'] for n in names]
f1s    = [results[n]['f1']       for n in names]

x = np.arange(len(names))
width = 0.35
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - width/2, accs, width, label='Accuracy', color='steelblue')
ax.bar(x + width/2, f1s,  width, label='F1 Score',  color='darkorange')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, ha='right')
ax.set_ylim(0, 1.1)
ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, 'model_comparison.png'), dpi=100)
plt.close()

# ── Best Model ─────────────────────────────────────────────────────────────────
best_name  = max(results, key=lambda n: results[n]['accuracy'])
best_model = results[best_name]['model']
print(f"\nBest Model: {best_name} ({results[best_name]['accuracy']:.4f})")

# Confusion matrix for best model
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(16, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix – {best_name}', fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, 'confusion_matrix.png'), dpi=100)
plt.close()

# Feature importance (Random Forest)
rf = results['Random Forest']['model']
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 5))
plt.bar(range(len(features)), importances[indices], color='teal', edgecolor='black')
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=15)
plt.title('Feature Importance – Random Forest', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, 'feature_importance.png'), dpi=100)
plt.close()

print("All charts saved.")

# ── Save Model & Artifacts ─────────────────────────────────────────────────────
joblib.dump(best_model, os.path.join(MODEL_DIR, 'model.pkl'))
joblib.dump(scaler,     os.path.join(MODEL_DIR, 'scaler.pkl'))
joblib.dump(le,         os.path.join(MODEL_DIR, 'label_encoder.pkl'))

# Save metrics for dashboard
metrics_df = pd.DataFrame([
    {'model': n, 'accuracy': v['accuracy'], 'precision': v['precision'],
     'recall': v['recall'], 'f1': v['f1']}
    for n, v in results.items()
])
metrics_df.to_csv(os.path.join(MODEL_DIR, 'metrics.csv'), index=False)

print("\nModel artifacts saved to /model/")
print("Training complete!")
