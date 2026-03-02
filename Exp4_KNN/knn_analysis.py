import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings, os

warnings.filterwarnings('ignore')
script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(script_dir, 'visualizations'), exist_ok=True)

print("=" * 60)
print("EXPERIMENT 4: K-NEAREST NEIGHBORS (KNN) CLASSIFICATION")
print("=" * 60)

print("\n1. LOADING DATA")
df = pd.read_csv(os.path.join(script_dir, 'Used_Car_Price_Prediction.csv'))
print(f"Dataset Shape: {df.shape}")
print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

print("\n2. CREATING PRICE CATEGORIES")
q33, q66 = df['sale_price'].quantile(0.33), df['sale_price'].quantile(0.66)
df['price_category'] = pd.cut(df['sale_price'], bins=[0, q33, q66, float('inf')], labels=['Low', 'Medium', 'High'])
print(f"Low: < ₹{q33:,.0f} | Medium: ₹{q33:,.0f}-₹{q66:,.0f} | High: > ₹{q66:,.0f}")
print(f"\nCategory Distribution:\n{df['price_category'].value_counts().sort_index()}")

print("\n3. FEATURE SELECTION & PREPROCESSING")
feature_cols = ['yr_mfr', 'kms_run', 'times_viewed', 'total_owners',
                'fuel_type', 'body_type', 'transmission', 'make', 'car_rating']
df_model = df[feature_cols + ['price_category']].copy()
df_model.fillna(df_model.mode().iloc[0], inplace=True)

categorical_cols = ['fuel_type', 'body_type', 'transmission', 'make', 'car_rating']
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    print(f"  Encoded '{col}': {len(le.classes_)} unique values")

X = df_model.drop('price_category', axis=1)
y = LabelEncoder().fit_transform(df_model['price_category'])
class_names = ['High', 'Low', 'Medium']
print(f"\nFeatures ({X.shape[1]}): {list(X.columns)}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

print("\n4. TRAIN-TEST SPLIT & FEATURE SCALING")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training: {len(X_train)} | Testing: {len(X_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Applied StandardScaler (mean=0, std=1)")

print("\n5. FINDING OPTIMAL K (Elbow Method)")
k_range = range(1, 31)
train_scores, test_scores, cv_scores_mean, cv_scores_std = [], [], [], []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    train_scores.append(accuracy_score(y_train, knn.predict(X_train_scaled)))
    test_scores.append(accuracy_score(y_test, knn.predict(X_test_scaled)))
    cv = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores_mean.append(cv.mean())
    cv_scores_std.append(cv.std())

best_k = list(k_range)[np.argmax(cv_scores_mean)]
print(f"Best K (by 5-fold CV): {best_k} → CV Accuracy: {max(cv_scores_mean)*100:.2f}%")
for k in [1, 3, 5, 7, 9, 11, 15, 21, 29]:
    i = k - 1
    print(f"  K={k:<3} Train: {train_scores[i]*100:.2f}% Test: {test_scores[i]*100:.2f}% CV: {cv_scores_mean[i]*100:.2f}% ± {cv_scores_std[i]*100:.2f}%")

print(f"\n6. KNN MODEL (K={best_k})")
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)
y_pred_train = knn_best.predict(X_train_scaled)
y_pred_test = knn_best.predict(X_test_scaled)
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
print(f"Training Accuracy: {train_acc*100:.2f}%")
print(f"Testing Accuracy:  {test_acc*100:.2f}%")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_test, target_names=class_names)}")

print("\n7. COMPARING DISTANCE METRICS")
metrics = ['euclidean', 'manhattan', 'minkowski']
metric_results = {}
print(f"{'Metric':<14} {'Train Acc':<12} {'Test Acc':<12} {'CV Acc':<12}")
for metric in metrics:
    knn_m = KNeighborsClassifier(n_neighbors=best_k, metric=metric)
    knn_m.fit(X_train_scaled, y_train)
    tr = accuracy_score(y_train, knn_m.predict(X_train_scaled))
    te = accuracy_score(y_test, knn_m.predict(X_test_scaled))
    cv = cross_val_score(knn_m, X_train_scaled, y_train, cv=5).mean()
    metric_results[metric] = {'train': tr, 'test': te, 'cv': cv}
    print(f"{metric:<14} {tr*100:<12.2f} {te*100:<12.2f} {cv*100:<12.2f}")

print("\n8. COMPARING WEIGHTED vs UNIFORM KNN")
weight_results = {}
for w in ['uniform', 'distance']:
    knn_w = KNeighborsClassifier(n_neighbors=best_k, weights=w)
    knn_w.fit(X_train_scaled, y_train)
    tr = accuracy_score(y_train, knn_w.predict(X_train_scaled))
    te = accuracy_score(y_test, knn_w.predict(X_test_scaled))
    cv = cross_val_score(knn_w, X_train_scaled, y_train, cv=5).mean()
    weight_results[w] = {'train': tr, 'test': te, 'cv': cv}
    print(f"Weights='{w}': Train={tr*100:.2f}% Test={te*100:.2f}% CV={cv*100:.2f}%")

print("\n9. GENERATING VISUALIZATIONS")
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Experiment 4: KNN Classification — Used Car Price Prediction', fontsize=15, fontweight='bold', y=1.01)

axes[0, 0].plot(k_range, [s*100 for s in train_scores], 'b-o', markersize=3, label='Train')
axes[0, 0].plot(k_range, [s*100 for s in test_scores], 'r-s', markersize=3, label='Test')
axes[0, 0].plot(k_range, [s*100 for s in cv_scores_mean], 'g-^', markersize=3, label='CV')
axes[0, 0].fill_between(k_range, [(m-s)*100 for m, s in zip(cv_scores_mean, cv_scores_std)],
                         [(m+s)*100 for m, s in zip(cv_scores_mean, cv_scores_std)], alpha=0.15, color='green')
axes[0, 0].axvline(x=best_k, color='gray', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
axes[0, 0].set(xlabel='K', ylabel='Accuracy (%)', title='Elbow Method — Optimal K')
axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3)

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_test), display_labels=class_names).plot(ax=axes[0, 1], cmap='Blues', colorbar=False)
axes[0, 1].set_title(f'Confusion Matrix (K={best_k})')

x_pos, width = np.arange(len(metrics)), 0.25
axes[0, 2].bar(x_pos - width, [metric_results[m]['train']*100 for m in metrics], width, label='Train', color='steelblue')
axes[0, 2].bar(x_pos, [metric_results[m]['test']*100 for m in metrics], width, label='Test', color='coral')
axes[0, 2].bar(x_pos + width, [metric_results[m]['cv']*100 for m in metrics], width, label='CV', color='seagreen')
axes[0, 2].set_xticks(x_pos); axes[0, 2].set_xticklabels([m.capitalize() for m in metrics])
axes[0, 2].set(ylabel='Accuracy (%)', title='Distance Metric Comparison')
axes[0, 2].legend(fontsize=8); axes[0, 2].grid(True, alpha=0.3, axis='y')

df['price_category'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 0],
    color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black')
axes[1, 0].set(title='Price Category Distribution', xlabel='Category', ylabel='Count')
axes[1, 0].tick_params(axis='x', rotation=0)

weight_names = list(weight_results.keys())
x_pos2 = np.arange(len(weight_names))
axes[1, 1].bar(x_pos2 - width, [weight_results[w]['train']*100 for w in weight_names], width, label='Train', color='steelblue')
axes[1, 1].bar(x_pos2, [weight_results[w]['test']*100 for w in weight_names], width, label='Test', color='coral')
axes[1, 1].bar(x_pos2 + width, [weight_results[w]['cv']*100 for w in weight_names], width, label='CV', color='seagreen')
axes[1, 1].set_xticks(x_pos2); axes[1, 1].set_xticklabels([w.capitalize() for w in weight_names])
axes[1, 1].set(ylabel='Accuracy (%)', title='Weight Function Comparison')
axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True, alpha=0.3, axis='y')

axes[1, 2].plot(k_range, [1 - s for s in test_scores], 'r-o', markersize=4)
axes[1, 2].axvline(x=best_k, color='gray', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
axes[1, 2].set(xlabel='K', ylabel='Error Rate', title='Error Rate vs K')
axes[1, 2].legend(fontsize=8); axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'visualizations/knn_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: visualizations/knn_results.png")

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
  - Feature Scaling: StandardScaler

Performance (K={best_k}):
  - Training Accuracy: {train_acc*100:.2f}%
  - Testing Accuracy:  {test_acc*100:.2f}%
""")
print("EXPERIMENT 4 COMPLETED!")
