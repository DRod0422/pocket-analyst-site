# utils.py

import pandas as pd
import numpy as np
import re

def clean_and_format_data(df_raw, log=False):
    """
    Cleans and formats raw data for analysis.
    
    Steps:
    - Removes empty rows/columns
    - Detects misplaced headers
    - Standardizes column names
    - Normalizes missing values
    - Converts numeric and date fields
    """
    df = df_raw.copy()
    change_log = []

    # Step 1: Drop fully empty rows/columns
    initial_shape = df.shape
    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    if df.shape != initial_shape:
        change_log.append("Removed empty rows/columns")

    # Step 2: Detect bad header row
    if df.iloc[0].isnull().mean() > 0.5 or any(df.iloc[0].astype(str).str.contains(r"report|date|company", case=False)):
        df.columns = df.iloc[1]
        df = df[2:].reset_index(drop=True)
        change_log.append("Adjusted header row (used row 2 as headers)")
    else:
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        change_log.append("Used row 1 as headers")

    # Step 3: Standardize column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(" ", "_")
    )
    # Handle duplicate columns
    from collections import Counter
    def deduplicate_columns(columns):
        counts = Counter()
        new_cols = []
        for col in columns:
            counts[col] += 1
            if counts[col] == 1:
                new_cols.append(col)
            else:
                new_cols.append(f"{col}.{counts[col]-1}")
        return new_cols
    
    df.columns = deduplicate_columns(df.columns)

    change_log.append("Standardized column names")

    # Step 4: Normalize missing values
    missing_vals = ["n/a", "na", "null", "none", "--", "-", ""]
    df.replace(missing_vals, np.nan, inplace=True)
    change_log.append("Standardized missing values")

    # Step 5: Convert numeric-looking strings
    for col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[\$,]", "", regex=True)  # remove currency symbols/commas
            .str.strip()
        )
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass  # ignore if conversion fails

    # Step 6: Try parsing dates
    for col in df.select_dtypes(include="object").columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > 0:
                df[col] = parsed
                change_log.append(f"Converted column '{col}' to datetime")
        except:
            continue

    if log:
        return df, change_log
    return df
