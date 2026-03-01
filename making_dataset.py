# Proggram to take the main colloms form each dataset 

import pandas as pd
import numpy as np

print("=" * 70)
print("CREATING main_dataset.csv")
print("=" * 70)

# columns for the final dataset
SELECTED_COLUMNS = ['cp', 'thalach', 'oldpeak', 'exang', 'chol', 'slope', 'age', 'sex', 'target']

# 1. Process heart_1.csv
df1 = pd.read_csv('heart_1.csv')

print("Creating the dataset main_dataset.csv")

# Rename columns
df1 = df1.rename(columns={
    'Age': 'age',
    'Sex': 'sex',
    'ChestPainType': 'cp',
    'MaxHR': 'thalach',
    'Oldpeak': 'oldpeak',
    'ExerciseAngina': 'exang',
    'Cholesterol': 'chol',
    'ST_Slope': 'slope',
    'HeartDisease': 'target'
})

# Convert categorical to numeric
df1['sex'] = df1['sex'].map({'M': 1, 'F': 0})
df1['cp'] = df1['cp'].map({'TA': 1, 'ATA': 2, 'NAP': 3, 'ASY': 4})
df1['exang'] = df1['exang'].map({'N': 0, 'Y': 1})
df1['slope'] = df1['slope'].map({'Up': 1, 'Flat': 2, 'Down': 3})

# Keep only selected columns
df1 = df1[SELECTED_COLUMNS]
print(f"   Extracted {len(df1)} rows")

# 2. Process heart_2.csv
print("[2/3] Reading heart_2.csv...")
df2 = pd.read_csv('heart_2.csv')

# Rename target column
df2 = df2.rename(columns={'target_binary': 'target'})

# Keep only selected columns
df2 = df2[SELECTED_COLUMNS]
print(f"   Extracted {len(df2)} rows")


# 3. Process heart_3.csv
print("[3/3] Reading heart_3.csv...")
df3 = pd.read_csv('heart_3.csv')

# Rename columns
df3 = df3.rename(columns={
    'thalch': 'thalach',
    'num': 'target'
})

# Convert categorical to numeric
df3['sex'] = df3['sex'].map({'Male': 1, 'Female': 0})
df3['cp'] = df3['cp'].map({
    'typical angina': 1,
    'atypical angina': 2,
    'non-anginal pain': 3,
    'asymptomatic': 4
})
df3['exang'] = df3['exang'].map({True: 1, False: 0})
df3['slope'] = df3['slope'].map({
    'upsloping': 1,
    'flat': 2,
    'downsloping': 3
})

# Convert target to binary (0-4 scale to 0-1)
df3['target'] = (df3['target'] > 0).astype(int)

# Keep only selected columns
df3 = df3[SELECTED_COLUMNS]
print(f"   Extracted {len(df3)} rows")

# ----------------------------------------------------------------------
# 4. Combine all datasets
# ----------------------------------------------------------------------
print("\n[4/5] Combining datasets...")

# Concatenate all three
combined = pd.concat([df1, df2, df3], ignore_index=True)
print(f"   Total rows: {len(combined)}")

# ----------------------------------------------------------------------
# 5. Fix cholesterol: Replace 0 with average
# ----------------------------------------------------------------------
print("\n[5/6] Fixing cholesterol values...")

# Count zeros in cholesterol
zero_count = (combined['chol'] == 0).sum()
print(f"   Found {zero_count} rows with cholesterol = 0")

if zero_count > 0:
    # Calculate average cholesterol (excluding zeros)
    avg_chol = combined[combined['chol'] > 0]['chol'].mean()
    print(f"   Average cholesterol (excluding 0s): {avg_chol:.1f} mg/dl")
    
    # Replace 0 with average
    combined.loc[combined['chol'] == 0, 'chol'] = round(avg_chol, 1)
    print(f"   ✓ Replaced {zero_count} zero values with {avg_chol:.1f}")
else:
    print("   ✓ No zero values found in cholesterol")

# Select best 306 rows from each dataset (least missing values)
combined['missing_count'] = combined.isna().sum(axis=1)
combined = combined.sort_values('missing_count')

# Take top 918 rows with least missing values
final_dataset = combined.head(918).drop('missing_count', axis=1)

# Shuffle the dataset
final_dataset = final_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
final_dataset.to_csv('main_dataset.csv', index=False)

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("✅ SUCCESS!")
print("=" * 70)
print(f"\nCreated: main_dataset.csv")
print(f"Rows: {len(final_dataset)}")
print(f"Columns: {len(final_dataset.columns)}")
print(f"\nColumns included:")
for i, col in enumerate(final_dataset.columns, 1):
    print(f"  {i}. {col}")

print(f"\nTarget distribution:")
print(f"  No disease (0): {(final_dataset['target'] == 0).sum()}")
print(f"  Has disease (1): {(final_dataset['target'] == 1).sum()}")

print(f"\nCholesterol statistics:")
print(f"  Min: {final_dataset['chol'].min():.1f} mg/dl")
print(f"  Average: {final_dataset['chol'].mean():.1f} mg/dl")
print(f"  Max: {final_dataset['chol'].max():.1f} mg/dl")
print(f"  Zero values: {(final_dataset['chol'] == 0).sum()}")

print("\n" + "=" * 70)
