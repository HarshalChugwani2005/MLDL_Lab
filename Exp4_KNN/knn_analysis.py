"""
Experiment 4: K-Nearest Neighbors (KNN) Classification
Dataset: Used Car Price Prediction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import warnings
import os

warnings.filterwarnings('ignore')

# Get script directory for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(script_dir, 'visualizations'), exist_ok=True)

# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("EXPERIMENT 4: K-NEAREST NEIGHBORS (KNN) CLASSIFICATION")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
print("\n1. LOADING DATA")
print("-" * 40)

df = pd.read_csv(os.path.join(script_dir, 'Used_Car_Price_Prediction.csv'))
print(f"Dataset Shape: {df.shape}")
print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ─────────────────────────────────────────────────────────────
print("\n2. CREATING PRICE CATEGORIES (Target Variable)")
print("-" * 40)

q33 = df['sale_price'].quantile(0.33)
q66 = df['sale_price'].quantile(0.66)

df['price_category'] = pd.cut(df['sale_price'],
                               bins=[0, q33, q66, float('inf')],
                               labels=['Low', 'Medium', 'High'])

print(f"Thresholds:")
print(f"  Low:    sale_price < ₹{q33:,.0f}")
print(f"  Medium: ₹{q33:,.0f} - ₹{q66:,.0f}")
print(f"  High:   sale_price > ₹{q66:,.0f}")
print(f"\nCategory Distribution:\n{df['price_category'].value_counts().sort_index()}")

# ─────────────────────────────────────────────────────────────
print("\n3. FEATURE SELECTION & PREPROCESSING")
print("-" * 40)

feature_cols = ['yr_mfr', 'kms_run', 'times_viewed', 'total_owners',
                'fuel_type', 'body_type', 'transmission', 'make', 'car_rating']

df_model = df[feature_cols + ['price_category']].copy()

# Fill missing values with mode
df_model.fillna(df_model.mode().iloc[0], inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['fuel_type', 'body_type', 'transmission', 'make', 'car_rating']

for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le
    print(f"  Encoded '{col}': {len(le.classes_)} unique values")

# Prepare X and y
X = df_model.drop('price_category', axis=1)
y = LabelEncoder().fit_transform(df_model['price_category'])
class_names = ['High', 'Low', 'Medium']

print(f"\nFeatures ({X.shape[1]}): {list(X.columns)}")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# ─────────────────────────────────────────────────────────────
print("\n4. TRAIN-TEST SPLIT & FEATURE SCALING")
print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# KNN is distance-based → Feature scaling is essential
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Applied StandardScaler (mean=0, std=1) — critical for KNN")

# ─────────────────────────────────────────────────────────────
print("\n5. FINDING OPTIMAL K (Elbow Method)")
print("-" * 40)

k_range = range(1, 31)
train_scores = []
test_scores = []
cv_scores_mean = []
cv_scores_std = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    train_scores.append(accuracy_score(y_train, knn.predict(X_train_scaled)))
    test_scores.append(accuracy_score(y_test, knn.predict(X_test_scaled)))
    cv = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores_mean.append(cv.mean())
    cv_scores_std.append(cv.std())

best_k = list(k_range)[np.argmax(cv_scores_mean)]
print(f"Best K (by 5-fold CV): {best_k}  →  CV Accuracy: {max(cv_scores_mean)*100:.2f}%")
print(f"\nK-value vs Accuracy (Test Set):")
for k in [1, 3, 5, 7, 9, 11, 15, 21, 29]:
    idx = k - 1
    print(f"  K={k:<3}  Train: {train_scores[idx]*100:.2f}%  "
          f"Test: {test_scores[idx]*100:.2f}%  "
          f"CV: {cv_scores_mean[idx]*100:.2f}% ± {cv_scores_std[idx]*100:.2f}%")

# ─────────────────────────────────────────────────────────────
print(f"\n6. KNN MODEL (K={best_k})")
print("-" * 40)

knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)

y_pred_train = knn_best.predict(X_train_scaled)
y_pred_test = knn_best.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy: {train_acc*100:.2f}%")
print(f"Testing Accuracy:  {test_acc*100:.2f}%")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_test, target_names=class_names)}")

# ─────────────────────────────────────────────────────────────
print("\n7. COMPARING DISTANCE METRICS")
print("-" * 40)

metrics = ['euclidean', 'manhattan', 'minkowski']
metric_results = {}

print(f"{'Metric':<14} {'Train Acc':<12} {'Test Acc':<12} {'CV Acc':<12}")
print("-" * 50)

for metric in metrics:
    knn_m = KNeighborsClassifier(n_neighbors=best_k, metric=metric)
    knn_m.fit(X_train_scaled, y_train)
    tr_acc = accuracy_score(y_train, knn_m.predict(X_train_scaled))
    te_acc = accuracy_score(y_test, knn_m.predict(X_test_scaled))
    cv_acc = cross_val_score(knn_m, X_train_scaled, y_train, cv=5).mean()
    metric_results[metric] = {'train': tr_acc, 'test': te_acc, 'cv': cv_acc}
    print(f"{metric:<14} {tr_acc*100:<12.2f} {te_acc*100:<12.2f} {cv_acc*100:<12.2f}")

# ─────────────────────────────────────────────────────────────
print("\n8. COMPARING WEIGHTED vs UNIFORM KNN")
print("-" * 40)

weight_results = {}
for w in ['uniform', 'distance']:
    knn_w = KNeighborsClassifier(n_neighbors=best_k, weights=w)
    knn_w.fit(X_train_scaled, y_train)
    tr_acc = accuracy_score(y_train, knn_w.predict(X_train_scaled))
    te_acc = accuracy_score(y_test, knn_w.predict(X_test_scaled))
    cv_acc = cross_val_score(knn_w, X_train_scaled, y_train, cv=5).mean()
    weight_results[w] = {'train': tr_acc, 'test': te_acc, 'cv': cv_acc}
    print(f"Weights='{w}':  Train={tr_acc*100:.2f}%  Test={te_acc*100:.2f}%  CV={cv_acc*100:.2f}%")

# ─────────────────────────────────────────────────────────────
print("\n9. GENERATING VISUALIZATIONS")
print("-" * 40)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Experiment 4: KNN Classification — Used Car Price Prediction',
             fontsize=15, fontweight='bold', y=1.01)

# ── Plot 1: Elbow Curve (K vs Accuracy) ──
ax1 = axes[0, 0]
ax1.plot(k_range, [s*100 for s in train_scores], 'b-o', markersize=3, label='Train Accuracy')
ax1.plot(k_range, [s*100 for s in test_scores], 'r-s', markersize=3, label='Test Accuracy')
ax1.plot(k_range, [s*100 for s in cv_scores_mean], 'g-^', markersize=3, label='CV Accuracy')
ax1.fill_between(k_range,
                 [(m - s)*100 for m, s in zip(cv_scores_mean, cv_scores_std)],
                 [(m + s)*100 for m, s in zip(cv_scores_mean, cv_scores_std)],
                 alpha=0.15, color='green')
ax1.axvline(x=best_k, color='gray', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
ax1.set_xlabel('K (Number of Neighbors)')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Elbow Method — Optimal K Selection')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# ── Plot 2: Confusion Matrix ──
ax2 = axes[0, 1]
cm = confusion_matrix(y_test, y_pred_test)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax2, cmap='Blues', colorbar=False)
ax2.set_title(f'Confusion Matrix (K={best_k})')

# ── Plot 3: Distance Metric Comparison ──
ax3 = axes[0, 2]
x_pos = np.arange(len(metrics))
width = 0.25
ax3.bar(x_pos - width, [metric_results[m]['train']*100 for m in metrics], width, label='Train', color='steelblue')
ax3.bar(x_pos, [metric_results[m]['test']*100 for m in metrics], width, label='Test', color='coral')
ax3.bar(x_pos + width, [metric_results[m]['cv']*100 for m in metrics], width, label='CV', color='seagreen')
ax3.set_xticks(x_pos)
ax3.set_xticklabels([m.capitalize() for m in metrics])
ax3.set_ylabel('Accuracy (%)')
ax3.set_title('Distance Metric Comparison')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis='y')

# ── Plot 4: Price Category Distribution ──
ax4 = axes[1, 0]
df['price_category'].value_counts().sort_index().plot(kind='bar', ax=ax4,
    color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black')
ax4.set_title('Price Category Distribution')
ax4.set_xlabel('Category')
ax4.set_ylabel('Count')
ax4.tick_params(axis='x', rotation=0)

# ── Plot 5: Uniform vs Distance Weights ──
ax5 = axes[1, 1]
weight_names = list(weight_results.keys())
x_pos2 = np.arange(len(weight_names))
ax5.bar(x_pos2 - width, [weight_results[w]['train']*100 for w in weight_names], width, label='Train', color='steelblue')
ax5.bar(x_pos2, [weight_results[w]['test']*100 for w in weight_names], width, label='Test', color='coral')
ax5.bar(x_pos2 + width, [weight_results[w]['cv']*100 for w in weight_names], width, label='CV', color='seagreen')
ax5.set_xticks(x_pos2)
ax5.set_xticklabels([w.capitalize() for w in weight_names])
ax5.set_ylabel('Accuracy (%)')
ax5.set_title('Weight Function Comparison')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3, axis='y')

# ── Plot 6: Error Rate vs K ──
ax6 = axes[1, 2]
error_rates = [1 - s for s in test_scores]
ax6.plot(k_range, error_rates, 'r-o', markersize=4)
ax6.axvline(x=best_k, color='gray', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
ax6.set_xlabel('K (Number of Neighbors)')
ax6.set_ylabel('Error Rate')
ax6.set_title('Error Rate vs K')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'visualizations/knn_results.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: visualizations/knn_results.png")

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

best_metric = max(metric_results, key=lambda m: metric_results[m]['cv'])
best_weight = max(weight_results, key=lambda w: weight_results[w]['cv'])

print(f"""
Dataset: Used Car Price Prediction ({len(df)} samples)
Task: Classify cars into Low / Medium / High price categories

KNN Configuration:
  - Optimal K: {best_k} (selected via 5-fold Cross-Validation)
  - Best Distance Metric: {best_metric}
  - Best Weight Function: {best_weight}
  - Feature Scaling: StandardScaler (required for distance-based KNN)

Performance (K={best_k}):
  - Training Accuracy: {train_acc*100:.2f}%
  - Testing Accuracy:  {test_acc*100:.2f}%

Key Insights:
  1. KNN is a lazy learner — no explicit training phase;
     classification is done at prediction time.
  2. Feature scaling is critical because KNN relies on distance
     calculations between data points.
  3. Small K → complex boundary (risk of overfitting)
     Large K → smoother boundary (risk of underfitting)
  4. The elbow method helps identify the K that balances bias
     and variance for optimal performance.
""")
print("EXPERIMENT 4 COMPLETED!")
