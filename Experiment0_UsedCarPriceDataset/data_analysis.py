import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('../Used_Car_Price_Prediction.csv')

# Exercise 1: NumPy Basics
# NumPy is the foundation library for numerical computing in Python.
# 1. Load sale_price as NumPy array
sale_price = data['sale_price'].to_numpy()

# 2. Compute mean, median, and standard deviation
mean_price = np.mean(sale_price)
median_price = np.median(sale_price)
std_dev_price = np.std(sale_price)
var_price = np.var(sale_price)

# 3. Perform Min-Max normalization
min_price = np.min(sale_price)
max_price = np.max(sale_price)
normalized_price = (sale_price - min_price) / (max_price - min_price)

# Add normalized price to dataframe
data['normalized_sale_price'] = normalized_price

print("="*60)
print("Exercise 1: NumPy Basics")
print("="*60)
print("\n2. Statistical Measures:")
print(f"   Mean:              {mean_price:,.2f}")
print(f"   Median:            {median_price:,.2f}")
print(f"   Standard Deviation: {std_dev_price:,.2f}")
print(f"   Variance:          {var_price:,.2f}")
print(f"   Min Value:         {min_price:,.2f}")
print(f"   Max Value:         {max_price:,.2f}")

print("\n3. Min-Max Normalization:")
print(f"   Formula: (X - Min) / (Max - Min)")
print(f"   Normalized Range: [0, 1]")
print(f"   Sample Normalized Values (first 5):")
for i in range(min(5, len(normalized_price))):
    print(f"      Original: {sale_price[i]:,.2f} → Normalized: {normalized_price[i]:.4f}")

# Exercise 2: Pandas Data Handling
# Pandas is used for data analysis and manipulation.
print("\nExercise 2: Pandas Data Handling")
print("Shape of the dataset:", data.shape)
print("Columns in the dataset:", data.columns)
print("Missing values in the dataset:\n", data.isnull().sum())

# Create Performance label based on sale_price
def performance_label(price):
    if price > 400000:
        return 'High'
    elif price >= 200000:
        return 'Medium'
    else:
        return 'Low'

data['Performance'] = data['sale_price'].apply(performance_label)
print("\nPerformance labels added to the dataset.")

# Exercise 3: Matplotlib Visualization
# Matplotlib is a low-level plotting library used to create graphs and charts.
# Line plot: kms_run vs sale_price
plt.figure(figsize=(8, 5))
plt.plot(data['kms_run'], data['sale_price'], marker='o', linestyle='-', color='b')
plt.title('Kms Run vs Sale Price')
plt.xlabel('Kms Run')
plt.ylabel('Sale Price')
plt.grid()
plt.savefig('visualizations/kms_run_vs_sale_price_lineplot.png')
plt.close()

# Histogram of sale_price
plt.figure(figsize=(8, 5))
plt.hist(data['sale_price'], bins=10, color='g', alpha=0.7)
plt.title('Histogram of Sale Price')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.grid()
plt.savefig('visualizations/histogram_sale_price.png')
plt.close()

# Exercise 4: Seaborn Visualization
# Seaborn is a high-level visualization library built on top of Matplotlib.
# Scatter plot: kms_run vs sale_price with fuel_type as hue
plt.figure(figsize=(8, 5))
sns.scatterplot(x='kms_run', y='sale_price', hue='fuel_type', data=data)
plt.title('Kms Run vs Sale Price')
plt.xlabel('Kms Run')
plt.ylabel('Sale Price')
plt.legend(title='Fuel Type')
plt.savefig('visualizations/kms_run_vs_sale_price_scatter.png')
plt.close()

# Heatmap for correlation analysis
plt.figure(figsize=(8, 5))
correlation_matrix = data[['kms_run', 'sale_price']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap of Correlation Matrix')
plt.savefig('visualizations/heatmap_correlation.png')
plt.close()

# Boxplot: Sale Price grouped by Fuel Type
plt.figure(figsize=(8, 5))
sns.boxplot(x='fuel_type', y='sale_price', data=data)
plt.title('Boxplot: Sale Price by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Sale Price')
plt.savefig('visualizations/boxplot_sale_price_by_fuel_type.png')
plt.close()