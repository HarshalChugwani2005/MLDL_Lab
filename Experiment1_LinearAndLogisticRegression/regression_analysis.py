"""
Experiment 1: Linear and Logistic Regression Implementation
Dataset: Used Car Price Prediction (Without Data Cleaning)
Author: MLDL Lab
Date: 2026

This script implements:
1. Linear Regression for Price Prediction
2. Logistic Regression for Classification
3. User Input for Price Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import os

# Create visualizations directory
os.makedirs('visualizations', exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================
# SECTION 1: DATA LOADING (NO CLEANING)
# ============================================================================

print("="*80)
print("EXPERIMENT 1: LINEAR AND LOGISTIC REGRESSION")
print("Dataset: Used Car Price Prediction (Without Data Cleaning)")
print("="*80)

# Load data
df = pd.read_csv('Used_Car_Price_Prediction.csv')

print("\n--- DATASET OVERVIEW ---")
print(f"Total Records: {len(df)}")
print(f"Total Features: {len(df.columns)}")

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- MISSING VALUES ---")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\n--- NUMERICAL STATISTICS ---")
print(df.describe())

# ============================================================================
# SECTION 2: MINIMAL PREPROCESSING (NO CLEANING)
# ============================================================================

print("\n" + "="*80)
print("SECTION 2: MINIMAL PREPROCESSING (NO DATA CLEANING)")
print("="*80)

# Create a copy for modeling
df_model = df.copy()

# Fill missing values with simple strategies (NOT cleaning, just making it usable)
# For numerical columns - fill with median
numerical_cols = df_model.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    df_model[col] = df_model[col].fillna(df_model[col].median())

# For categorical columns - fill with mode
categorical_cols = df_model.select_dtypes(include=['object', 'bool']).columns
for col in categorical_cols:
    if df_model[col].isnull().sum() > 0:
        df_model[col] = df_model[col].fillna(df_model[col].mode()[0] if len(df_model[col].mode()) > 0 else 'unknown')

# Encode categorical variables
label_encoders = {}

# Encode fuel_type
le_fuel = LabelEncoder()
df_model['fuel_type_encoded'] = le_fuel.fit_transform(df_model['fuel_type'].astype(str))
label_encoders['fuel_type'] = le_fuel

# Encode body_type
le_body = LabelEncoder()
df_model['body_type_encoded'] = le_body.fit_transform(df_model['body_type'].astype(str))
label_encoders['body_type'] = le_body

# Encode transmission
le_trans = LabelEncoder()
df_model['transmission_encoded'] = le_trans.fit_transform(df_model['transmission'].astype(str))
label_encoders['transmission'] = le_trans

# Encode make (manufacturer)
le_make = LabelEncoder()
df_model['make_encoded'] = le_make.fit_transform(df_model['make'].astype(str))
label_encoders['make'] = le_make

# Convert boolean to int
bool_cols = ['assured_buy', 'is_hot', 'reserved', 'warranty_avail']
for col in bool_cols:
    df_model[col] = df_model[col].astype(bool).astype(int)

# Calculate car age
current_year = 2021  # Reference year from dataset
df_model['car_age'] = current_year - df_model['yr_mfr']

print(f"Records for modeling: {len(df_model)}")
print("Encoding complete for categorical variables")

# ============================================================================
# SECTION 3: LINEAR REGRESSION - PRICE PREDICTION
# ============================================================================

print("\n" + "="*80)
print("SECTION 3: LINEAR REGRESSION - PRICE PREDICTION")
print("="*80)

# Select features for Linear Regression
linear_features = [
    'yr_mfr', 'kms_run', 'car_age', 'total_owners',
    'fuel_type_encoded', 'body_type_encoded', 'transmission_encoded',
    'make_encoded', 'assured_buy', 'is_hot'
]

# Prepare data
X_linear = df_model[linear_features].copy()
y_linear = df_model['sale_price'].copy()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_linear, y_linear, test_size=0.2, random_state=42
)

# Scale features
scaler_linear = StandardScaler()
X_train_scaled = scaler_linear.fit_transform(X_train)
X_test_scaled = scaler_linear.transform(X_test)

# Train Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = linear_model.predict(X_train_scaled)
y_pred_test = linear_model.predict(X_test_scaled)

# Calculate metrics
print("\n--- LINEAR REGRESSION RESULTS ---")
print(f"\nTraining Set:")
print(f"  R² Score: {r2_score(y_train, y_pred_train):.4f}")
print(f"  MAE: ₹{mean_absolute_error(y_train, y_pred_train):,.2f}")
print(f"  RMSE: ₹{np.sqrt(mean_squared_error(y_train, y_pred_train)):,.2f}")

print(f"\nTest Set:")
print(f"  R² Score: {r2_score(y_test, y_pred_test):.4f}")
print(f"  MAE: ₹{mean_absolute_error(y_test, y_pred_test):,.2f}")
print(f"  RMSE: ₹{np.sqrt(mean_squared_error(y_test, y_pred_test)):,.2f}")

# Feature Coefficients
print("\n--- FEATURE COEFFICIENTS ---")
coef_df = pd.DataFrame({
    'Feature': linear_features,
    'Coefficient': linear_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)
print(coef_df.to_string(index=False))

# ============================================================================
# SECTION 4: LOGISTIC REGRESSION - CLASSIFICATION
# ============================================================================

print("\n" + "="*80)
print("SECTION 4: LOGISTIC REGRESSION - CLASSIFICATION")
print("="*80)

# Create binary target: Premium (above median) vs Budget (below median)
median_price = df_model['sale_price'].median()
df_model['is_premium'] = (df_model['sale_price'] > median_price).astype(int)

print(f"\nMedian Price (Threshold): ₹{median_price:,.2f}")
print(f"Premium Cars (Above Median): {df_model['is_premium'].sum()}")
print(f"Budget Cars (Below Median): {len(df_model) - df_model['is_premium'].sum()}")

# Prepare data for Logistic Regression
logistic_features = [
    'yr_mfr', 'kms_run', 'car_age', 'total_owners',
    'fuel_type_encoded', 'body_type_encoded', 'transmission_encoded',
    'make_encoded', 'assured_buy', 'is_hot'
]

X_logistic = df_model[logistic_features].copy()
y_logistic = df_model['is_premium'].copy()

# Split data
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X_logistic, y_logistic, test_size=0.2, random_state=42, stratify=y_logistic
)

# Scale features
scaler_logistic = StandardScaler()
X_train_log_scaled = scaler_logistic.fit_transform(X_train_log)
X_test_log_scaled = scaler_logistic.transform(X_test_log)

# Train Logistic Regression Model
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train_log_scaled, y_train_log)

# Predictions
y_pred_log = logistic_model.predict(X_test_log_scaled)
y_pred_proba = logistic_model.predict_proba(X_test_log_scaled)[:, 1]

# Calculate metrics
print("\n--- LOGISTIC REGRESSION RESULTS ---")
print(f"\nClassification Metrics:")
print(f"  Accuracy:  {accuracy_score(y_test_log, y_pred_log):.4f}")
print(f"  Precision: {precision_score(y_test_log, y_pred_log):.4f}")
print(f"  Recall:    {recall_score(y_test_log, y_pred_log):.4f}")
print(f"  F1 Score:  {f1_score(y_test_log, y_pred_log):.4f}")
print(f"  ROC AUC:   {roc_auc_score(y_test_log, y_pred_proba):.4f}")

print("\n--- CONFUSION MATRIX ---")
cm = confusion_matrix(y_test_log, y_pred_log)
print(f"                 Predicted")
print(f"              Budget  Premium")
print(f"Actual Budget   {cm[0,0]:5d}    {cm[0,1]:5d}")
print(f"Actual Premium  {cm[1,0]:5d}    {cm[1,1]:5d}")

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test_log, y_pred_log, target_names=['Budget', 'Premium']))

# ============================================================================
# SECTION 5: VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("SECTION 5: GENERATING VISUALIZATIONS")
print("="*80)

# Visualization 1: Actual vs Predicted (Linear Regression)
print("Creating Visualization 1: Actual vs Predicted...")
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(y_test, y_pred_test, alpha=0.5, edgecolors='none', s=30)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
ax.set_title('Linear Regression: Actual vs Predicted Sale Price', fontsize=14, fontweight='bold')
ax.set_xlabel('Actual Price (₹)', fontsize=12)
ax.set_ylabel('Predicted Price (₹)', fontsize=12)
ax.legend()
ax.ticklabel_format(style='plain', axis='both')
plt.tight_layout()
plt.savefig('visualizations/01_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()

# Visualization 2: Feature Coefficients
print("Creating Visualization 2: Feature Coefficients...")
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#FF6B6B' if x < 0 else '#4ECDC4' for x in coef_df['Coefficient']]
ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
ax.set_title('Linear Regression: Feature Coefficients', fontsize=14, fontweight='bold')
ax.set_xlabel('Coefficient Value', fontsize=12)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig('visualizations/02_feature_coefficients.png', dpi=150, bbox_inches='tight')
plt.close()

# Visualization 3: Residual Plot
print("Creating Visualization 3: Residual Plot...")
residuals = y_test - y_pred_test
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_pred_test, residuals, alpha=0.5, s=20)
axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0].set_title('Residuals vs Predicted Values', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted Price (₹)')
axes[0].set_ylabel('Residuals (₹)')

axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Residuals (₹)')
axes[1].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('visualizations/03_residual_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# Visualization 4: Confusion Matrix
print("Creating Visualization 4: Confusion Matrix...")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Budget', 'Premium'], yticklabels=['Budget', 'Premium'])
ax.set_title('Logistic Regression: Confusion Matrix', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/04_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# Visualization 5: Price Distribution
print("Creating Visualization 5: Price Distribution...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df_model['sale_price'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(x=median_price, color='r', linestyle='--', lw=2, label=f'Median: ₹{median_price:,.0f}')
ax.set_title('Sale Price Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Sale Price (₹)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.legend()
ax.ticklabel_format(style='plain', axis='x')
plt.tight_layout()
plt.savefig('visualizations/05_price_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Visualization 6: Price by Manufacturer
print("Creating Visualization 6: Price by Manufacturer...")
fig, ax = plt.subplots(figsize=(12, 6))
top_makes = df_model['make'].value_counts().head(8).index
df_top = df_model[df_model['make'].isin(top_makes)]
sns.boxplot(x='make', y='sale_price', data=df_top, palette='husl', ax=ax)
ax.set_title('Sale Price Distribution by Manufacturer', fontsize=14, fontweight='bold')
ax.set_xlabel('Manufacturer', fontsize=12)
ax.set_ylabel('Sale Price (₹)', fontsize=12)
ax.ticklabel_format(style='plain', axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/06_price_by_manufacturer.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll visualizations saved to 'visualizations/' folder!")

# ============================================================================
# SECTION 6: USER INPUT FOR PRICE PREDICTION
# ============================================================================

print("\n" + "="*80)
print("SECTION 6: USER INPUT FOR PRICE PREDICTION")
print("="*80)

def get_user_input_and_predict():
    """
    Get user input and predict car price using the trained model
    """
    print("\n" + "-"*60)
    print("USED CAR PRICE PREDICTION SYSTEM")
    print("-"*60)
    
    # Display available options
    print("\n--- Available Options ---")
    print(f"Fuel Types: {list(le_fuel.classes_)}")
    print(f"Body Types: {list(le_body.classes_)}")
    print(f"Transmission: {list(le_trans.classes_)}")
    print(f"Manufacturers: {list(le_make.classes_)}")
    
    try:
        print("\n--- Enter Car Details ---")
        
        # Year of manufacture
        yr_mfr = int(input("Year of Manufacture (e.g., 2015): "))
        
        # Kilometers run
        kms_run = int(input("Kilometers Run (e.g., 50000): "))
        
        # Total owners
        total_owners = int(input("Total Owners (e.g., 1, 2, 3): "))
        
        # Fuel type
        print(f"\nFuel Types: {list(le_fuel.classes_)}")
        fuel_type = input("Fuel Type: ").strip().lower()
        
        # Body type
        print(f"\nBody Types: {list(le_body.classes_)}")
        body_type = input("Body Type: ").strip().lower()
        
        # Transmission
        print(f"\nTransmission: {list(le_trans.classes_)}")
        transmission = input("Transmission: ").strip().lower()
        
        # Manufacturer
        print(f"\nManufacturers: {list(le_make.classes_)}")
        make = input("Manufacturer: ").strip().lower()
        
        # Boolean features
        assured_buy = input("Assured Buy? (yes/no): ").strip().lower() == 'yes'
        is_hot = input("Is Hot Deal? (yes/no): ").strip().lower() == 'yes'
        
        # Calculate car age
        car_age = current_year - yr_mfr
        
        # Encode categorical variables
        try:
            fuel_encoded = le_fuel.transform([fuel_type])[0]
        except:
            print(f"Warning: Unknown fuel type '{fuel_type}'. Using default.")
            fuel_encoded = 0
            
        try:
            body_encoded = le_body.transform([body_type])[0]
        except:
            print(f"Warning: Unknown body type '{body_type}'. Using default.")
            body_encoded = 0
            
        try:
            trans_encoded = le_trans.transform([transmission])[0]
        except:
            print(f"Warning: Unknown transmission '{transmission}'. Using default.")
            trans_encoded = 0
            
        try:
            make_encoded = le_make.transform([make])[0]
        except:
            print(f"Warning: Unknown manufacturer '{make}'. Using default.")
            make_encoded = 0
        
        # Create feature array
        user_features = np.array([[
            yr_mfr, kms_run, car_age, total_owners,
            fuel_encoded, body_encoded, trans_encoded,
            make_encoded, int(assured_buy), int(is_hot)
        ]])
        
        # Scale features
        user_features_scaled = scaler_linear.transform(user_features)
        
        # Predict price
        predicted_price = linear_model.predict(user_features_scaled)[0]
        
        # Also predict if it's premium or budget
        user_features_log_scaled = scaler_logistic.transform(user_features)
        is_premium_pred = logistic_model.predict(user_features_log_scaled)[0]
        premium_prob = logistic_model.predict_proba(user_features_log_scaled)[0][1]
        
        # Display results
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"\n🚗 Car Details Summary:")
        print(f"   • Year: {yr_mfr}")
        print(f"   • Kilometers: {kms_run:,}")
        print(f"   • Car Age: {car_age} years")
        print(f"   • Owners: {total_owners}")
        print(f"   • Fuel: {fuel_type}")
        print(f"   • Body: {body_type}")
        print(f"   • Transmission: {transmission}")
        print(f"   • Manufacturer: {make}")
        
        print(f"\n💰 PREDICTED SALE PRICE: ₹{predicted_price:,.2f}")
        
        category = "PREMIUM" if is_premium_pred else "BUDGET"
        print(f"\n📊 Category: {category} (Confidence: {premium_prob*100:.1f}%)")
        print(f"   (Threshold: ₹{median_price:,.2f})")
        
        # Price range estimate
        mae = mean_absolute_error(y_test, y_pred_test)
        print(f"\n📈 Estimated Price Range:")
        print(f"   • Lower: ₹{max(0, predicted_price - mae):,.2f}")
        print(f"   • Upper: ₹{predicted_price + mae:,.2f}")
        
        return predicted_price
        
    except ValueError as e:
        print(f"\n❌ Error: Invalid input. Please enter valid values.")
        print(f"   Details: {e}")
        return None
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

# ============================================================================
# SECTION 7: SUMMARY AND RESULTS
# ============================================================================

print("\n" + "="*80)
print("SECTION 7: SUMMARY")
print("="*80)

print("\n--- MODEL PERFORMANCE SUMMARY ---")
print("\n📈 LINEAR REGRESSION (Price Prediction):")
print(f"   • R² Score (Test): {r2_score(y_test, y_pred_test):.4f}")
print(f"   • MAE: ₹{mean_absolute_error(y_test, y_pred_test):,.2f}")
print(f"   • RMSE: ₹{np.sqrt(mean_squared_error(y_test, y_pred_test)):,.2f}")

print("\n📊 LOGISTIC REGRESSION (Premium/Budget Classification):")
print(f"   • Accuracy: {accuracy_score(y_test_log, y_pred_log):.4f}")
print(f"   • F1 Score: {f1_score(y_test_log, y_pred_log):.4f}")
print(f"   • ROC AUC: {roc_auc_score(y_test_log, y_pred_proba):.4f}")

print("\n--- FILES GENERATED ---")
print("   • visualizations/01_actual_vs_predicted.png")
print("   • visualizations/02_feature_coefficients.png")
print("   • visualizations/03_residual_analysis.png")
print("   • visualizations/04_confusion_matrix.png")
print("   • visualizations/05_price_distribution.png")
print("   • visualizations/06_price_by_manufacturer.png")

# ============================================================================
# MAIN EXECUTION - USER INTERACTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("INTERACTIVE PRICE PREDICTION")
    print("="*80)
    
    while True:
        print("\n--- OPTIONS ---")
        print("1. Predict Car Price (Enter your car details)")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == '1':
            get_user_input_and_predict()
        elif choice == '2':
            print("\n✅ Thank you for using the Used Car Price Prediction System!")
            print("="*80)
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
