# utils.py

import numpy as np
import pandas as pd

def clean_and_format_data(df, log=False):
    clean_log = []
    df_clean = df.copy()

    # 1. Strip whitespace from column names
    df_clean.columns = df_clean.columns.str.strip()
    clean_log.append("Stripped whitespace from column names")

    # 2. Normalize missing values
    df_clean.replace(["", " ", "--", "n/a", "N/A", "NA", "null"], np.nan, inplace=True)
    clean_log.append("Standardized common missing value indicators")

    # 3. Remove fully empty rows
    before = len(df_clean)
    df_clean.dropna(how='all', inplace=True)
    after = len(df_clean)
    clean_log.append(f"Removed {before - after} completely empty rows")

    # 4. Detect and drop fully empty or unnamed columns
    empty_cols = df_clean.columns[df_clean.isna().all()].tolist()
    unnamed_cols = [col for col in df_clean.columns if "unnamed" in col.lower()]
    drop_cols = list(set(empty_cols + unnamed_cols))
    if drop_cols:
        df_clean.drop(columns=drop_cols, inplace=True)
        clean_log.append(f"Dropped columns: {', '.join(drop_cols)}")

    # 5. Attempt to convert datetime columns *before* numeric
    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            try:
                converted = pd.to_datetime(df_clean[col], errors="coerce", infer_datetime_format=True)
                if converted.notna().sum() > 0.5 * len(df_clean):
                    df_clean[col] = converted
                    clean_log.append(f"Safely converted column '{col}' to datetime")
            except Exception:
                pass

    # 6. Attempt to convert remaining object columns to numeric
    for col in df_clean.columns:
        # Count how many non-null values can be safely converted to numbers
        num_convertible = pd.to_numeric(df_clean[col], errors='coerce').notnull().sum()
        total_nonnull = df_clean[col].notnull().sum()
    
        # Only convert if at least 90% of non-null values are numeric
        if total_nonnull > 0 and (num_convertible / total_nonnull) >= 0.9:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            clean_log.append(f"Converted column '{col}' to numeric (90%+ values numeric)")
        else:
            df_clean[col] = df_clean[col].astype(str).str.strip()
            clean_log.append(f"Kept column '{col}' as text (mixed or mostly non-numeric)")
    
        return df_clean, clean_log if log else df_clean

