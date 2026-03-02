import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings, os, time

warnings.filterwarnings('ignore')
script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(script_dir, 'visualizations'), exist_ok=True)

print("EXPERIMENT 5: SUPPORT VECTOR MACHINE (SVM) CLASSIFICATION")

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

print("\n5. SVM WITH DIFFERENT KERNELS")
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
kernel_results = {}

print(f"{'Kernel':<12} {'Train Acc':<12} {'Test Acc':<12} {'CV Acc':<12} {'Time (s)':<10}")
print("-" * 58)
for kernel in kernels:
    start = time.time()
    svm = SVC(kernel=kernel, random_state=42)
    svm.fit(X_train_scaled, y_train)
    tr = accuracy_score(y_train, svm.predict(X_train_scaled))
    te = accuracy_score(y_test, svm.predict(X_test_scaled))
    cv = cross_val_score(svm, X_train_scaled, y_train, cv=5).mean()
    elapsed = time.time() - start
    kernel_results[kernel] = {'model': svm, 'train': tr, 'test': te, 'cv': cv, 'time': elapsed}
    print(f"{kernel:<12} {tr*100:<12.2f} {te*100:<12.2f} {cv*100:<12.2f} {elapsed:<10.2f}")

best_kernel = max(kernel_results, key=lambda k: kernel_results[k]['cv'])
print(f"\nBest Kernel: {best_kernel} (CV Accuracy: {kernel_results[best_kernel]['cv']*100:.2f}%)")

print("\n6. HYPERPARAMETER TUNING (GridSearchCV)")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf', 'linear', 'poly']
}

print(f"Search Space: {len(param_grid['C'])} x {len(param_grid['gamma'])} x {len(param_grid['kernel'])} = "
      f"{len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['kernel'])} combinations x 5-fold CV")

start = time.time()
grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy',
                           n_jobs=-1, verbose=0, return_train_score=True)
grid_search.fit(X_train_scaled, y_train)
grid_time = time.time() - start

print(f"\nGridSearchCV completed in {grid_time:.2f}s")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Accuracy: {grid_search.best_score_*100:.2f}%")

results_df = pd.DataFrame(grid_search.cv_results_)
print(f"\nTop 5 Parameter Combinations:")
top5 = results_df.nsmallest(5, 'rank_test_score')[['params', 'mean_test_score', 'std_test_score', 'mean_train_score']]
for _, row in top5.iterrows():
    print(f"  {row['params']} → CV: {row['mean_test_score']*100:.2f}% ± {row['std_test_score']*100:.2f}%")

print("\n7. BEST SVM MODEL (Tuned)")
best_svm = grid_search.best_estimator_
y_pred_train = best_svm.predict(X_train_scaled)
y_pred_test = best_svm.predict(X_test_scaled)
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy: {train_acc*100:.2f}%")
print(f"Testing Accuracy:  {test_acc*100:.2f}%")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_test, target_names=class_names)}")

print("\n8. EFFECT OF C (Regularization Parameter)")
C_values = [0.01, 0.1, 1, 10, 100]
c_results = {}
for c in C_values:
    svm_c = SVC(C=c, kernel=grid_search.best_params_['kernel'], gamma=grid_search.best_params_['gamma'], random_state=42)
    svm_c.fit(X_train_scaled, y_train)
    tr = accuracy_score(y_train, svm_c.predict(X_train_scaled))
    te = accuracy_score(y_test, svm_c.predict(X_test_scaled))
    c_results[c] = {'train': tr, 'test': te}
    print(f"  C={c:<6} Train: {tr*100:.2f}%  Test: {te*100:.2f}%")

print("\n9. DEFAULT vs TUNED MODEL COMPARISON")
default_svm = SVC(random_state=42)
default_svm.fit(X_train_scaled, y_train)
default_train = accuracy_score(y_train, default_svm.predict(X_train_scaled))
default_test = accuracy_score(y_test, default_svm.predict(X_test_scaled))
default_cv = cross_val_score(default_svm, X_train_scaled, y_train, cv=5).mean()
tuned_cv = grid_search.best_score_

print(f"{'Model':<18} {'Train Acc':<12} {'Test Acc':<12} {'CV Acc':<12}")
print("-" * 54)
print(f"{'Default SVM':<18} {default_train*100:<12.2f} {default_test*100:<12.2f} {default_cv*100:<12.2f}")
print(f"{'Tuned SVM':<18} {train_acc*100:<12.2f} {test_acc*100:<12.2f} {tuned_cv*100:<12.2f}")
improvement = (test_acc - default_test) * 100
print(f"\nImprovement after tuning: {improvement:+.2f}% (test accuracy)")

print("\n10. GENERATING VISUALIZATIONS")
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Experiment 5: SVM Classification — Used Car Price Prediction', fontsize=15, fontweight='bold', y=1.01)

x_pos, width = np.arange(len(kernels)), 0.25
axes[0, 0].bar(x_pos - width, [kernel_results[k]['train']*100 for k in kernels], width, label='Train', color='steelblue')
axes[0, 0].bar(x_pos, [kernel_results[k]['test']*100 for k in kernels], width, label='Test', color='coral')
axes[0, 0].bar(x_pos + width, [kernel_results[k]['cv']*100 for k in kernels], width, label='CV', color='seagreen')
axes[0, 0].set_xticks(x_pos); axes[0, 0].set_xticklabels([k.capitalize() for k in kernels])
axes[0, 0].set(ylabel='Accuracy (%)', title='Kernel Comparison')
axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3, axis='y')

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_test), display_labels=class_names).plot(ax=axes[0, 1], cmap='Blues', colorbar=False)
axes[0, 1].set_title(f'Confusion Matrix (Tuned SVM)')

c_vals = list(c_results.keys())
axes[0, 2].plot(range(len(c_vals)), [c_results[c]['train']*100 for c in c_vals], 'b-o', label='Train')
axes[0, 2].plot(range(len(c_vals)), [c_results[c]['test']*100 for c in c_vals], 'r-s', label='Test')
axes[0, 2].set_xticks(range(len(c_vals))); axes[0, 2].set_xticklabels(c_vals)
axes[0, 2].set(xlabel='C (Regularization)', ylabel='Accuracy (%)', title='Effect of C Parameter')
axes[0, 2].legend(fontsize=8); axes[0, 2].grid(True, alpha=0.3)

df['price_category'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 0],
    color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black')
axes[1, 0].set(title='Price Category Distribution', xlabel='Category', ylabel='Count')
axes[1, 0].tick_params(axis='x', rotation=0)

models = ['Default SVM', 'Tuned SVM']
axes[1, 1].bar(np.arange(2) - 0.15, [default_train*100, train_acc*100], 0.3, label='Train', color='steelblue')
axes[1, 1].bar(np.arange(2) + 0.15, [default_test*100, test_acc*100], 0.3, label='Test', color='coral')
axes[1, 1].set_xticks(np.arange(2)); axes[1, 1].set_xticklabels(models)
axes[1, 1].set(ylabel='Accuracy (%)', title='Default vs Tuned SVM')
axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True, alpha=0.3, axis='y')

pivot = results_df[results_df['param_kernel'] == grid_search.best_params_['kernel']].copy()
if 'param_C' in pivot.columns and 'param_gamma' in pivot.columns:
    gamma_vals = sorted(pivot['param_gamma'].unique(), key=str)
    c_unique = sorted(pivot['param_C'].unique())
    for gv in gamma_vals:
        subset = pivot[pivot['param_gamma'] == gv].sort_values('param_C')
        axes[1, 2].plot(range(len(c_unique)), subset['mean_test_score'].values * 100, '-o', label=f'γ={gv}', markersize=5)
    axes[1, 2].set_xticks(range(len(c_unique))); axes[1, 2].set_xticklabels(c_unique)
    axes[1, 2].set(xlabel='C', ylabel='CV Accuracy (%)', title=f'GridSearch Heatmap ({grid_search.best_params_["kernel"]})')
    axes[1, 2].legend(fontsize=7, title='Gamma'); axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'visualizations/svm_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: visualizations/svm_results.png")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
Dataset: Used Car Price Prediction ({len(df)} samples)
Task: Classify cars into Low / Medium / High price categories

SVM Configuration:
  - Best Kernel: {grid_search.best_params_['kernel']}
  - Best C: {grid_search.best_params_['C']}
  - Best Gamma: {grid_search.best_params_['gamma']}
  - Feature Scaling: StandardScaler
  - Hyperparameter Tuning: GridSearchCV (5-fold CV)

Performance (Tuned SVM):
  - Training Accuracy: {train_acc*100:.2f}%
  - Testing Accuracy:  {test_acc*100:.2f}%
  - Improvement over default: {improvement:+.2f}%
""")
print("EXPERIMENT 5 COMPLETED!")
