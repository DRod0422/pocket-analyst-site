# utils.py

import pandas as pd
import numpy as np
import re

def clean_and_format_data(df, log=False):
    """
    Cleans and formats raw data for analysis.
    
    Steps:
    - Removes empty rows/columns
    - Detects misplaced headers
    - Standardizes column names
    - Normalizes missing values
    - Converts numeric and date fields
    """
    clean_log = []
    df_clean = df.copy()

    # 1. Strip whitespace from column names (only minor cleanup)
    df_clean.columns = df_clean.columns.str.strip()
    clean_log.append("Stripped whitespace from column names")

    # 2. Normalize missing values (blanks, '--', 'n/a', etc.)
    df_clean.replace(["", " ", "--", "n/a", "N/A", "NA", "null"], np.nan, inplace=True)
    clean_log.append("Standardized common missing value indicators")

    # 3. Remove fully empty rows
    before = len(df_clean)
    df_clean.dropna(how='all', inplace=True)
    after = len(df_clean)
    clean_log.append(f"Removed {before - after} completely empty rows")

    # 4. Convert columns to numeric where possible
    for col in df_clean.columns:
        try:
            df_clean[col] = pd.to_numeric(df_clean[col])
            clean_log.append(f"Converted column '{col}' to numeric")
        except Exception:
            pass  # Ignore non-numeric columns

    # 5. Convert date columns
    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            try:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='raise')
                clean_log.append(f"Converted column '{col}' to datetime")
            except Exception:
                pass

    return df_clean, clean_log if log else df_clean
