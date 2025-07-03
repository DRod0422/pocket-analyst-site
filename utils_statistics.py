#utils/statistics.py

import pandas as pd
import scipy.stats as stats

def run_auto_statistical_insights(df):
    results = []

    # --- One-Sample T-Test (against 0) ---
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 10:
            t_stat, p_val = stats.ttest_1samp(series, 0)
            if p_val < 0.05:
                results.append(f"📌 **{col}** has a mean significantly different from 0 (p = {p_val:.4f}).")

    # --- Two-Sample T-Test ---
    cat_cols = df.select_dtypes(include="object").columns
    for cat_col in cat_cols:
        unique_vals = df[cat_col].dropna().unique()
        if len(unique_vals) == 2:
            for num_col in numeric_cols:
                group1 = df[df[cat_col] == unique_vals[0]][num_col].dropna()
                group2 = df[df[cat_col] == unique_vals[1]][num_col].dropna()
                if len(group1) > 5 and len(group2) > 5:
                    t_stat, p_val = stats.ttest_ind(group1, group2)
                    if p_val < 0.05:
                        results.append(f"🔍 **{num_col}** differs significantly between **{unique_vals[0]}** and **{unique_vals[1]}** (p = {p_val:.4f}).")

    # --- Chi-Square Tests ---
    for i in range(len(cat_cols)):
        for j in range(i + 1, len(cat_cols)):
            tbl = pd.crosstab(df[cat_cols[i]], df[cat_cols[j]])
            if tbl.shape[0] > 1 and tbl.shape[1] > 1:
                chi2, p_val, _, _ = stats.chi2_contingency(tbl)
                if p_val < 0.05:
                    results.append(f"⚠️ **{cat_cols[i]}** and **{cat_cols[j]}** appear dependent (Chi² p = {p_val:.4f}).")

    # --- Correlation Scan ---
    corr_matrix = df[numeric_cols].corr(method="pearson")
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                results.append(f"🔗 **{numeric_cols[i]}** and **{numeric_cols[j]}** are strongly correlated (r = {corr_val:.2f}).")

    return results
