import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings, os, time

# Setup
warnings.filterwarnings('ignore')
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

viz_dir = os.path.join(script_dir, 'visualizations')
os.makedirs(viz_dir, exist_ok=True)

print("EXPERIMENT 5: ADVANCED SVM CLASSIFICATION & VISUALIZATION")

# 1. LOADING DATA
# Assuming the CSV is in the same directory
df = pd.read_csv(os.path.join(script_dir, 'Used_Car_Price_Prediction.csv'))

# 2. CREATING PRICE CATEGORIES
q33, q66 = df['sale_price'].quantile(0.33), df['sale_price'].quantile(0.66)
df['price_category'] = pd.cut(df['sale_price'], bins=[0, q33, q66, float('inf')], labels=['Low', 'Medium', 'High'])

# 3. FEATURE SELECTION & PREPROCESSING
feature_cols = ['yr_mfr', 'kms_run', 'times_viewed', 'total_owners', 'fuel_type', 'body_type', 'transmission', 'make', 'car_rating']
df_model = df[feature_cols + ['price_category']].copy()
df_model.fillna(df_model.mode().iloc[0], inplace=True)

# Encode Features
le = LabelEncoder()
for col in ['fuel_type', 'body_type', 'transmission', 'make', 'car_rating']:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

X = df_model.drop('price_category', axis=1)
# Encode Target (Alphabetical: 0=High, 1=Low, 2=Medium)
y = le.fit_transform(df_model['price_category'])
class_names = le.classes_ 

# 4. TRAIN-TEST SPLIT & SCALING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. KERNEL COMPARISON
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
kernel_results = {}
for kernel in kernels:
    svm = SVC(kernel=kernel, random_state=42)
    svm.fit(X_train_scaled, y_train)
    cv_acc = cross_val_score(svm, X_train_scaled, y_train, cv=5).mean()
    kernel_results[kernel] = cv_acc

# 6. HYPERPARAMETER TUNING
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf'] # Focused on RBF for the heatmap visualization
}
grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
best_svm = grid_search.best_estimator_

# 7. FINAL VISUALIZATION SUITE
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3)
fig.suptitle('SVM Analysis: Used Car Price Prediction', fontsize=20, fontweight='bold', y=0.95)

# Plot 1: Kernel Comparison
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(kernel_results.keys(), [v*100 for v in kernel_results.values()], color='skyblue', edgecolor='black')
ax1.set_title('Accuracy by Kernel Type')
ax1.set_ylabel('CV Accuracy (%)')

# Plot 2: Confusion Matrix
ax2 = fig.add_subplot(gs[0, 1])
ConfusionMatrixDisplay.from_estimator(best_svm, X_test_scaled, y_test, display_labels=class_names, cmap='Blues', ax=ax2, colorbar=False)
ax2.set_title('Confusion Matrix (Tuned Model)')

# Plot 3: Grid Search Heatmap (C vs Gamma)
ax3 = fig.add_subplot(gs[0, 2])
res_df = pd.DataFrame(grid_search.cv_results_)
pivot = res_df.pivot(index='param_C', columns='param_gamma', values='mean_test_score')
sns.heatmap(pivot, annot=True, cmap='YlGnBu', ax=ax3)
ax3.set_title('GridSearch: C vs Gamma (RBF)')

# Plot 4: PCA Decision Boundaries (The "Better Visualization")
# We reduce features to 2D to actually SEE the SVM boundaries
ax4 = fig.add_subplot(gs[1, :2])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)
svm_2d = SVC(kernel=grid_search.best_params_['kernel'], C=grid_search.best_params_['C']).fit(X_pca, y_train)

h = .05 # step size in mesh
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax4.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
scatter = ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, edgecolors='k', cmap='viridis', s=30)
ax4.set_title(f"Decision Boundaries (PCA Reduced Space - {grid_search.best_params_['kernel']} kernel)")
legend1 = ax4.legend(*scatter.legend_elements(), title="Categories")
ax4.add_artist(legend1)

# Plot 5: Feature Importance (If Linear) or Feature Distribution
ax5 = fig.add_subplot(gs[1, 2])
df['price_category'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'], ax=ax5)
ax5.set_title('Target Class Balance')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(viz_dir, 'svm_final_report.png'), dpi=150)
print(f"Success! Visualization saved to: {viz_dir}/svm_final_report.png")

# Final Results Output
print(f"\nBEST PARAMETERS: {grid_search.best_params_}")
print(f"FINAL TEST ACCURACY: {accuracy_score(y_test, best_svm.predict(X_test_scaled))*100:.2f}%")