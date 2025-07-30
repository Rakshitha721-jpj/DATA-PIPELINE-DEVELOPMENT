import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

file_path = "sample_data.csv"

# ✅ Step 1: Load dataset safely
if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    df = pd.DataFrame()  # Empty DataFrame to prevent crash
else:
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        df = pd.DataFrame()

# ✅ Step 2: Clean data only if DataFrame is not empty
if df.empty:
    print("❌ No data loaded. Check your CSV file.")
else:
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    if df.empty:
        print("⚠️ CSV has no rows after cleaning.")
    else:
        dtypes_orig = df.dtypes.values
        if len(dtypes_orig) > 0:
            dtype_orig = np.result_type(*dtypes_orig)
            print("Combined dtype:", dtype_orig)

        # ✅ Step 3: Encode categorical columns
        le = LabelEncoder()
        for col in ['Gender', 'Department']:
            if col in df.columns:
                df[col] = le.fit_transform(df[col])

        # ✅ Step 4: Scale numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # ✅ Step 5: Save the cleaned and processed data
        df.to_csv("cleaned_data.csv", index=False)

        print("✅ Data pipeline completed! File saved as cleaned_data.csv.")
        print("Shape:", df.shape)
        print(df.head())
