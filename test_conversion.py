import pandas as pd
import numpy as np

df = pd.read_csv('Exp9_RNN/AirQualityUCI.csv', delimiter=';')
print('Before conversion:')
print(f'T dtype: {df["T"].dtype}')
print(f'T sample: {df["T"].head(5).tolist()}')

# Apply conversion like in my script
df["T"] = df["T"].astype(str).str.replace(',', '.')
df["T"] = pd.to_numeric(df["T"], errors='coerce')

print('\nAfter conversion:')
print(f'T dtype: {df["T"].dtype}')
print(f'T sample: {df["T"].head(5).tolist()}')
print(f'T null count: {df["T"].isnull().sum()}')
print(f'T non-null count: {df["T"].notna().sum()}')
print(f'T values where present: {df[df["T"].notna()]["T"].head(10).tolist()}')
