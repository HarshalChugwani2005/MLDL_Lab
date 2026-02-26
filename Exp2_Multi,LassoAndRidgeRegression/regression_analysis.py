"""
Experiment 2: Multiple Linear, Lasso, and Ridge Regression
Dataset: Used Car Price Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import os

warnings.filterwarnings('ignore')

# Get script directory for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(script_dir, 'visualizations'), exist_ok=True)

print("EXPERIMENT 2: MULTIPLE LINEAR, LASSO, AND RIDGE REGRESSION")

# 1. LOAD DATA
print("\n1. LOADING DATA")
df = pd.read_csv(os.path.join(script_dir, 'Used_Car_Price_Prediction.csv'))
print(f"Dataset Shape: {df.shape}")

# 2. PREPARE FEATURES
print("\n2. PREPARING FEATURES")

# Select features (excluding price-derived columns to avoid data leakage)
numerical_features = ['yr_mfr', 'kms_run', 'times_viewed', 'total_owners']
categorical_features = ['fuel_type', 'body_type', 'transmission', 'make']
target = 'sale_price'

df_work = df.copy()

# Fill missing values
for col in numerical_features:
    df_work[col] = df_work[col].fillna(df_work[col].median())

for col in categorical_features:
    df_work[col] = df_work[col].fillna(df_work[col].mode()[0])

# Encode categorical features
for col in categorical_features:
    le = LabelEncoder()
    df_work[col + '_encoded'] = le.fit_transform(df_work[col].astype(str))

# Prepare X and y
feature_cols = numerical_features + [col + '_encoded' for col in categorical_features]
X = df_work[feature_cols]
y = df_work[target]

print(f"Features: {feature_cols}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# 3. TRAIN-TEST SPLIT
print("\n3. SPLITTING DATA")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. TRAIN MODELS
print("\n4. TRAINING MODELS")

models = {
    'Linear Regression': LinearRegression(),
    'Lasso (α=1.0)': Lasso(alpha=1.0, random_state=42),
    'Ridge (α=1.0)': Ridge(alpha=1.0, random_state=42)
}

results = {}

for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    results[name] = {
        'model': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'y_test_pred': y_test_pred
    }
    
    print(f"\n{name}:")
    print(f"  Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
    print(f"  Train RMSE: ₹{train_rmse:,.0f} | Test RMSE: ₹{test_rmse:,.0f}")

# 5. MODEL COMPARISON
print("\n5. MODEL COMPARISON")
print(f"{'Model':<20} {'Train R²':<12} {'Test R²':<12} {'Test RMSE':<15}")
for name, res in results.items():
    print(f"{name:<20} {res['train_r2']:<12.4f} {res['test_r2']:<12.4f} ₹{res['test_rmse']:<14,.0f}")

# Find best model
best_model = max(results, key=lambda x: results[x]['test_r2'])
print(f"\nBest Model: {best_model} (Test R² = {results[best_model]['test_r2']:.4f})")

# 6. FEATURE COEFFICIENTS
print("\n6. FEATURE COEFFICIENTS")
coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Linear': results['Linear Regression']['model'].coef_,
    'Lasso': results['Lasso (α=1.0)']['model'].coef_,
    'Ridge': results['Ridge (α=1.0)']['model'].coef_
})
print(coef_df.to_string(index=False))

# 7. VISUALIZATIONS
print("\n7. GENERATING VISUALIZATIONS")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Model Comparison (R² Scores)
ax1 = axes[0, 0]
model_names = list(results.keys())
x = np.arange(len(model_names))
width = 0.35
ax1.bar(x - width/2, [results[m]['train_r2'] for m in model_names], width, label='Train R²', color='#3498db')
ax1.bar(x + width/2, [results[m]['test_r2'] for m in model_names], width, label='Test R²', color='#e74c3c')
ax1.set_ylabel('R² Score')
ax1.set_title('R² Score Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=15)
ax1.legend()
ax1.set_ylim(0, 1)

# Plot 2: Actual vs Predicted (Best Model)
ax2 = axes[0, 1]
y_pred_best = results[best_model]['y_test_pred']
ax2.scatter(y_test, y_pred_best, alpha=0.5, color='#3498db', s=20)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Price (₹)')
ax2.set_ylabel('Predicted Price (₹)')
ax2.set_title(f'Actual vs Predicted ({best_model})')
ax2.legend()

# Plot 3: Feature Coefficients
ax3 = axes[1, 0]
coef_df['Abs_Linear'] = np.abs(coef_df['Linear'])
coef_sorted = coef_df.sort_values('Abs_Linear', ascending=True)
y_pos = np.arange(len(feature_cols))
ax3.barh(y_pos, coef_sorted['Linear'], color='#3498db', alpha=0.8, label='Linear')
ax3.set_yticks(y_pos)
ax3.set_yticklabels(coef_sorted['Feature'])
ax3.set_xlabel('Coefficient Value')
ax3.set_title('Feature Coefficients (Linear Regression)')
ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

# Plot 4: Residual Distribution
ax4 = axes[1, 1]
residuals = y_test.values - y_pred_best
ax4.hist(residuals, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Residuals (₹)')
ax4.set_ylabel('Frequency')
ax4.set_title(f'Residual Distribution ({best_model})')

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'visualizations/regression_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: visualizations/regression_results.png")

# 8. SUMMARY
print("\nSUMMARY")
print(f"""
Dataset: Used Car Price Prediction ({len(df)} samples)
Task: Predict car sale price using regression

Results:
- Linear Regression: R² = {results['Linear Regression']['test_r2']:.4f}
- Lasso Regression:  R² = {results['Lasso (α=1.0)']['test_r2']:.4f}
- Ridge Regression:  R² = {results['Ridge (α=1.0)']['test_r2']:.4f}

Best Model: {best_model}

Key Insights:
1. Linear Regression provides baseline performance
2. Lasso performs feature selection (shrinks some coefficients to zero)
3. Ridge shrinks coefficients but keeps all features
4. All models explain ~{results[best_model]['test_r2']*100:.0f}% of price variance

Top Predictive Features:
{coef_df.nlargest(3, 'Abs_Linear')[['Feature', 'Linear']].to_string(index=False)}
""")
print("EXPERIMENT COMPLETED!")
