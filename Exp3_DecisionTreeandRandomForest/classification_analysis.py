"""
Experiment 3: Decision Tree and Random Forest Classification
Dataset: Used Car Price Prediction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
import os

warnings.filterwarnings('ignore')

# Get script directory for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(script_dir, 'visualizations'), exist_ok=True)

print("EXPERIMENT 3: DECISION TREE AND RANDOM FOREST")

print("\n1. LOADING DATA")

df = pd.read_csv(os.path.join(script_dir, 'Used_Car_Price_Prediction.csv'))
print(f"Dataset Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

print("\n2. CREATING PRICE CATEGORIES")

q33 = df['sale_price'].quantile(0.33)
q66 = df['sale_price'].quantile(0.66)

df['price_category'] = pd.cut(df['sale_price'], 
                               bins=[0, q33, q66, float('inf')],
                               labels=['Low', 'Medium', 'High'])

print(f"Low: < ₹{q33:,.0f}")
print(f"Medium: ₹{q33:,.0f} - ₹{q66:,.0f}")
print(f"High: > ₹{q66:,.0f}")
print(f"\nCategory Distribution:\n{df['price_category'].value_counts()}")

print("\n3. PREPARING FEATURES")

feature_cols = ['yr_mfr', 'kms_run', 'times_viewed', 'total_owners',
                'fuel_type', 'body_type', 'transmission', 'make', 'car_rating']

df_model = df[feature_cols + ['price_category']].copy()

# Fill missing values
df_model.fillna(df_model.mode().iloc[0], inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['fuel_type', 'body_type', 'transmission', 'make', 'car_rating']

for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le

# Prepare X and y
X = df_model.drop('price_category', axis=1)
y = LabelEncoder().fit_transform(df_model['price_category'])

print(f"Features: {list(X.columns)}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

print("\n4. SPLITTING DATA")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

print("\n5. DECISION TREE CLASSIFIER")

# Train Decision Tree
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Predictions
y_pred_dt = dt_model.predict(X_test)

# Evaluate
dt_train_acc = accuracy_score(y_train, dt_model.predict(X_train))
dt_test_acc = accuracy_score(y_test, y_pred_dt)

print(f"Training Accuracy: {dt_train_acc*100:.2f}%")
print(f"Testing Accuracy: {dt_test_acc*100:.2f}%")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_dt, target_names=['High', 'Low', 'Medium'])}")

print("\n6. RANDOM FOREST CLASSIFIER")

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate
rf_train_acc = accuracy_score(y_train, rf_model.predict(X_train))
rf_test_acc = accuracy_score(y_test, y_pred_rf)

print(f"Training Accuracy: {rf_train_acc*100:.2f}%")
print(f"Testing Accuracy: {rf_test_acc*100:.2f}%")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_rf, target_names=['High', 'Low', 'Medium'])}")

print("\n7. MODEL COMPARISON")

print(f"{'Model':<20} {'Train Acc':<12} {'Test Acc':<12}")
print(f"{'Decision Tree':<20} {dt_train_acc*100:<12.2f} {dt_test_acc*100:<12.2f}")
print(f"{'Random Forest':<20} {rf_train_acc*100:<12.2f} {rf_test_acc*100:<12.2f}")

print("\n8. FEATURE IMPORTANCE (Random Forest)")

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(importance_df.to_string(index=False))

print("\n9. GENERATING VISUALIZATIONS")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Decision Tree Structure
plt.subplot(2, 2, 1)
plot_tree(dt_model, feature_names=X.columns.tolist(), 
          class_names=['High', 'Low', 'Medium'], filled=True, 
          rounded=True, fontsize=6, max_depth=3)
plt.title('Decision Tree Structure')

# Plot 2: Confusion Matrix - Decision Tree
axes[0, 1].set_title('Confusion Matrix - Decision Tree')
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_dt), 
                       display_labels=['High', 'Low', 'Medium']).plot(ax=axes[0, 1], cmap='Blues')

# Plot 3: Confusion Matrix - Random Forest
axes[1, 0].set_title('Confusion Matrix - Random Forest')
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf), 
                       display_labels=['High', 'Low', 'Medium']).plot(ax=axes[1, 0], cmap='Greens')

# Plot 4: Feature Importance
axes[1, 1].barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
axes[1, 1].set_xlabel('Importance')
axes[1, 1].set_title('Feature Importance (Random Forest)')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'visualizations/classification_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: visualizations/classification_results.png")

print("\nSUMMARY")
print(f"""
Dataset: Used Car Price Prediction ({len(df)} samples)
Task: Classify cars into Low/Medium/High price categories

Results:
- Decision Tree: {dt_test_acc*100:.2f}% accuracy
- Random Forest: {rf_test_acc*100:.2f}% accuracy

Best Model: {'Random Forest' if rf_test_acc > dt_test_acc else 'Decision Tree'}

Key Insight: Random Forest performs better by combining multiple 
decision trees (ensemble learning) to reduce overfitting.

Top 3 Important Features:
1. {importance_df.iloc[0]['Feature']} ({importance_df.iloc[0]['Importance']:.3f})
2. {importance_df.iloc[1]['Feature']} ({importance_df.iloc[1]['Importance']:.3f})
3. {importance_df.iloc[2]['Feature']} ({importance_df.iloc[2]['Importance']:.3f})
""")
print("EXPERIMENT COMPLETED!")
