import pandas as pd
import numpy as np

df = pd.read_csv('Exp9_RNN/AirQualityUCI.csv', delimiter=';')

# Replace commas with dots in all columns
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.replace(',', '.')

# Convert numeric columns
numeric_cols = df.columns[2:]
for col in numeric_cols:
    if col in df.columns and col != 'Time':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Replace -200
df = df.replace(-200, np.nan)

# Drop unnamed
df = df.loc[:, ~df.columns.str.contains('Unnamed')]

print("Missing values after conversion and -200 replacement:")
print(df.isnull().sum())
print("\n\nTotal rows:", len(df))
print("\nColumns with NO missing values:")
print(df.columns[df.isnull().sum() == 0].tolist())
