import pandas as pd

# ---------------------------------------------------------
# STEP 1: Load the integrated dataset produced by Module 6
# ---------------------------------------------------------

df = pd.read_csv("data/processed/master_wheat_dataset.csv")

# Basic structure
print("Dataset shape:", df.shape)

print("\nColumns:")
for i, column in enumerate(df.columns, start=1):
    print(f"{i:3}. {column}")

# Data types
print("\nData types:")
print(df.dtypes.value_counts())

# Missing values
missing = df.isna().sum()
missing = missing[missing > 0]

print("\nColumns containing missing values:")
print(missing)

# Duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())