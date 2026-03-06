# WinBert AI — Pocket Analyst
# Revamped: Local Llama 3 via Ollama | No OpenAI | Full AI narration throughout

import streamlit as st
import pandas as pd
import numpy as np
import io
import datetime
import base64
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from utils import clean_and_format_data
import scipy.stats as stats
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
from pathlib import Path
from PIL import Image

# ─────────────────────────────────────────────
# LOCAL LLM CONFIG
# ─────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3"  # confirm with: ollama list

def ask_winbert(system_prompt: str, user_prompt: str, timeout: int = 120) -> str:
    """Call local Llama 3 via Ollama. Returns response text."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        "stream": False
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        return "⚠️ WinBert is offline. Make sure Ollama is running: `ollama serve`"
    except Exception as e:
        return f"⚠️ WinBert error: {e}"


def winbert_chat_turn(messages: list, timeout: int = 120) -> str:
    """Multi-turn chat — pass full message history."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        return "⚠️ WinBert is offline. Make sure Ollama is running: `ollama serve`"
    except Exception as e:
        return f"⚠️ WinBert error: {e}"


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="WinBert AI | Pocket Analyst",
    layout="wide",
    page_icon="🤖"
)

logo_path = Path("assets/bmlogo.png")
st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 12px;">
        <img src="https://raw.githubusercontent.com/DRod0422/pocket-analyst-site/main/{logo_path}"
             alt="WinBert Logo" style="height: 40px;">
        <h1 style="margin: 0;">WinBert AI | Pocket Analyst</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("Upload your data. Ask questions. Get AI-powered insights — no coding required.")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Upload & Clean",
    "🤖 WinBert Chat",
    "📊 Quick Analysis",
    "📈 Forecasting",
    "📐 Data Science Tools"
])


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def detect_chart_type_and_columns(question, df):
    q = question.lower()
    col_match = lambda keyword: next((col for col in df.columns if keyword in col.lower()), None)
    if "bar chart" in q:
        col = next((col for col in df.columns if col.lower() in q), None)
        return ("bar", col)
    elif "line chart" in q:
        x_col = col_match("date") or col_match("time")
        y_col = col_match("price") or col_match("value")
        return ("line", (x_col, y_col))
    elif "scatter" in q:
        x_col = col_match("sqft") or col_match("square")
        y_col = col_match("price")
        color_col = col_match("bed") or col_match("type")
        return ("scatter", (x_col, y_col, color_col))
    return (None, None)


def run_auto_statistical_insights(df):
    results = []
    numeric_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include="object").columns

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 10:
            t_stat, p_val = stats.ttest_1samp(series, 0)
            if p_val < 0.05:
                results.append(f"📌 **{col}** has a mean significantly different from 0 (p = {p_val:.4f}).")

    for cat_col in cat_cols:
        unique_vals = df[cat_col].dropna().unique()
        if len(unique_vals) == 2:
            for num_col in numeric_cols:
                g1 = df[df[cat_col] == unique_vals[0]][num_col].dropna()
                g2 = df[df[cat_col] == unique_vals[1]][num_col].dropna()
                if len(g1) > 5 and len(g2) > 5:
                    _, p_val = stats.ttest_ind(g1, g2)
                    if p_val < 0.05:
                        results.append(f"🔍 **{num_col}** differs significantly between **{unique_vals[0]}** and **{unique_vals[1]}** (p = {p_val:.4f}).")

    for i in range(len(cat_cols)):
        for j in range(i + 1, len(cat_cols)):
            tbl = pd.crosstab(df[cat_cols[i]], df[cat_cols[j]])
            if tbl.shape[0] > 1 and tbl.shape[1] > 1:
                chi2, p_val, _, _ = stats.chi2_contingency(tbl)
                if p_val < 0.05:
                    results.append(f"⚠️ **{cat_cols[i]}** and **{cat_cols[j]}** appear dependent (Chi² p = {p_val:.4f}).")

    corr_matrix = df[numeric_cols].corr(method="pearson")
    nc = list(numeric_cols)
    for i in range(len(nc)):
        for j in range(i + 1, len(nc)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                results.append(f"🔗 **{nc[i]}** and **{nc[j]}** are strongly correlated (r = {corr_val:.2f}).")

    return results


def xbar_r_chart(df, column, subgroup_size=5, show_insight=False):
    subgroups = [df[column].iloc[i:i+subgroup_size] for i in range(0, len(df[column]), subgroup_size)]
    subgroups = [g for g in subgroups if len(g) == subgroup_size]
    means  = [g.mean() for g in subgroups]
    ranges = [g.max() - g.min() for g in subgroups]
    xbar = np.mean(means)
    rbar = np.mean(ranges)
    UCLx = xbar + (2.66 * rbar)
    LCLx = xbar - (2.66 * rbar)
    fig, ax = plt.subplots()
    ax.plot(means, marker='o', label='Subgroup Means')
    ax.axhline(xbar,  color='green', label='Center Line')
    ax.axhline(UCLx, color='red', linestyle='--', label='UCL')
    ax.axhline(LCLx, color='red', linestyle='--', label='LCL')
    ax.set_title(f'X̄ Chart – {column}')
    ax.set_xlabel('Subgroup')
    ax.set_ylabel(column)
    ax.legend()
    st.pyplot(fig)
    if show_insight:
        stable = all(LCLx <= m <= UCLx for m in means)
        insight = f"""
**💡 WinBert Insight on `{column}`:**
- Process average: **{xbar:.2f}**
- Range average: **{rbar:.2f}**
- Control limits: UCL = **{UCLx:.2f}**, LCL = **{LCLx:.2f}**
- {"✅ Process appears stable." if stable else "❌ Out-of-control conditions detected."}
        """
        st.markdown(insight)
        b64 = base64.b64encode(insight.encode()).decode()
        st.markdown(f'<a href="data:file/txt;base64,{b64}" download="WinBert_SPC_Insight.txt">📥 Download Insight</a>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 1 — UPLOAD & CLEAN
# ═══════════════════════════════════════════════════════════
with tab1:
    uploaded_file = st.file_uploader(
        "Upload your dataset (.csv or .xlsx)", type=["csv", "xlsx"], key="main_upload"
    )

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        else:
            xls = pd.ExcelFile(uploaded_file)
            sheet_names = xls.sheet_names
            selected_sheet = st.selectbox("Select a sheet to load", sheet_names)
            df_raw = pd.read_excel(xls, sheet_name=selected_sheet)
            st.session_state["selected_sheet"] = selected_sheet

        df_raw = df_raw.loc[:, ~df_raw.columns.str.contains("^Unnamed", na=False)]
        df_raw = df_raw.dropna(axis=1, how="all")
        st.session_state["df_raw"] = df_raw
        st.session_state["last_uploaded_name"] = uploaded_file.name

        use_cleaning = st.checkbox("🧼 Auto-clean uploaded data?", value=st.session_state.get("use_cleaning", False))
        st.session_state["use_cleaning"] = use_cleaning

        if use_cleaning:
            df_clean, clean_log = clean_and_format_data(df_raw, log=True)
            st.session_state["df_clean"] = df_clean
            df_current = df_clean
            st.success("✅ File cleaned and loaded.")
            for entry in clean_log:
                st.markdown(f"🧼 {entry}")
        else:
            st.session_state["df_clean"] = None
            df_current = df_raw

        st.session_state["df_current"] = df_current
        st.session_state["df_current_full"] = df_current

        if len(df_current) > 5000:
            df_sample = df_current.sample(n=1000, random_state=42)
            st.warning(f"Large dataset ({len(df_current)} rows). Sampling 1,000 rows for UI performance.")
        else:
            df_sample = df_current
        st.session_state["df_sample"] = df_sample

        st.info(f"Loaded: **{df_current.shape[0]}** rows × **{df_current.shape[1]}** columns")
        st.subheader("Preview")
        st.dataframe(df_current.head(100))

        # ── Normalization ──────────────────────────────────────
        st.markdown("---")
        st.markdown("## 🧹 Data Normalization & Encoding")
        df_clean_state = st.session_state.get("df_clean")
        drop_columns = []
        if df_clean_state is not None:
            drop_columns = st.multiselect("Optional: Drop columns before processing", df_clean_state.columns.tolist())
        else:
            st.warning("⚠️ Enable 'Auto-clean' above to unlock normalization.")

        scaler_choice = st.selectbox("Normalization method:", ("MinMaxScaler", "StandardScaler", "RobustScaler"))
        skip_scaling  = st.checkbox("⚠️ Skip normalization (data is already scaled)")

        if st.button("⚙️ Normalize & Encode"):
            from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
            df_encoded = df_clean_state.drop(columns=drop_columns) if drop_columns else df_clean_state.copy()
            df_encoded = pd.get_dummies(df_encoded, drop_first=True)
            numeric_cols = df_encoded.select_dtypes(include=["int64", "float64"]).columns
            if not skip_scaling and len(numeric_cols) > 0:
                scaler = {"MinMaxScaler": MinMaxScaler(), "StandardScaler": StandardScaler(), "RobustScaler": RobustScaler()}[scaler_choice]
                df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
            st.session_state["normalized_data"] = df_encoded
            st.success("✅ Dataset normalized and encoded!")
            st.dataframe(df_encoded.head())

        # ── WinBert Auto-Insights on Upload ───────────────────
        st.markdown("---")
        st.markdown("## ✨ WinBert Auto-Insights")
        st.markdown("WinBert reads your data and tells you what it sees — in plain English.")

        if "ai_ran_once" not in st.session_state:
            st.session_state.ai_ran_once = False

        if st.session_state.ai_ran_once:
            st.success("✅ WinBert already analyzed this dataset. Switch to the **WinBert Chat** tab to keep exploring.")
        else:
            if st.button("🔍 Run WinBert Analysis"):
                with st.spinner("WinBert is reading your data..."):
                    csv_snippet = df_current.to_csv(index=False)[:4000]
                    numeric_summary = df_current.describe().to_string()[:2000]
                    missing = df_current.isnull().sum()
                    missing_summary = missing[missing > 0].to_string() if missing.any() else "No missing values."

                    system = "You are WinBert, a friendly expert data analyst. You give clear, plain-English insights. Never use jargon. Talk directly to the user."
                    user = f"""
Here is a dataset. Give me 4–5 specific insights about it. Point out:
- What the data appears to be about
- Any columns that look interesting or unusual
- Missing data issues I should know about
- One business question this data could help answer

Missing data:
{missing_summary}

Statistical summary:
{numeric_summary}

Sample rows:
{csv_snippet}
"""
                    with st.expander("✨ WinBert's First Look", expanded=True):
                        answer = ask_winbert(system, user)
                        st.markdown(answer)
                        st.session_state.ai_ran_once = True
                        # Pre-seed the chat with this context
                        st.session_state["chat_history"] = [
                            {"role": "system", "content": system},
                            {"role": "user",   "content": user},
                            {"role": "assistant", "content": answer}
                        ]
                        st.info("💬 Head to the **WinBert Chat** tab to keep asking questions.")


# ═══════════════════════════════════════════════════════════
# TAB 2 — WINBERT CHAT (multi-turn, no question limit)
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🤖 WinBert AI Assistant")
    st.markdown("Ask WinBert anything about your data. This is a full conversation — WinBert remembers what you've discussed.")

    df_clean  = st.session_state.get("df_clean")
    df_raw    = st.session_state.get("df_raw")
    df_current = df_clean if df_clean is not None else df_raw

    if df_current is None:
        st.warning("⚠️ Upload a dataset in Tab 1 first.")
        st.stop()

    st.session_state["df_current"] = df_current

    sample_option = st.checkbox("Use full dataset for analysis (may be slower for large files)", value=False)
    if not sample_option and len(df_current) > 5000:
        df_sample = df_current.sample(n=1000, random_state=42)
        st.caption(f"Sampling 1,000 of {len(df_current)} rows for speed.")
    else:
        df_sample = df_current
    st.session_state["df_sample"] = df_sample

    # Initialize chat history if needed
    if "chat_history" not in st.session_state:
        csv_snippet     = df_sample.head(10).to_string(index=False)
        numeric_summary = df_sample.describe().to_string()[:1500]
        st.session_state["chat_history"] = [
            {
                "role": "system",
                "content": (
                    f"You are WinBert, an expert data analyst. "
                    f"The user has uploaded a dataset with {df_current.shape[0]} rows and {df_current.shape[1]} columns. "
                    f"Here is a preview:\n{csv_snippet}\n\nStatistical summary:\n{numeric_summary}\n\n"
                    "Answer questions about this data clearly and concisely. "
                    "If the user asks for a chart, describe what it would show. "
                    "Be direct. Avoid jargon unless the user seems technical."
                )
            }
        ]

    # Render chat history
    with st.expander("💬 Example questions to ask WinBert", expanded=False):
        st.markdown("""
- What are the key trends in this dataset?
- Which column has the most missing values?
- What is the average value of [column]?
- Which month or region had the highest revenue?
- What's unusual or surprising about this data?
- What kind of model would work best for predicting [column]?
        """)

    # Display conversation
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_question = st.chat_input("Ask WinBert about your data...")

    if user_question:
        st.session_state["chat_history"].append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("WinBert is thinking..."):
                answer = winbert_chat_turn(st.session_state["chat_history"])
            st.markdown(answer)
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})

        # Auto chart rendering based on question keywords
        chart_type, chart_cols = detect_chart_type_and_columns(user_question, df_sample)
        if chart_type == "bar" and chart_cols and chart_cols in df_sample.columns:
            st.plotly_chart(px.bar(df_sample, x=chart_cols))
        elif chart_type == "line" and chart_cols and all(chart_cols):
            st.plotly_chart(px.line(df_sample, x=chart_cols[0], y=chart_cols[1]))
        elif chart_type == "scatter" and chart_cols and all(chart_cols):
            st.plotly_chart(px.scatter(df_sample, x=chart_cols[0], y=chart_cols[1], color=chart_cols[2]))

    if st.button("🗑️ Clear Chat History"):
        st.session_state.pop("chat_history", None)
        st.rerun()


# ═══════════════════════════════════════════════════════════
# TAB 3 — QUICK ANALYSIS
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 📊 Smart Auto Insights")
    st.markdown("Instant overview of your dataset — summaries, stats, and visuals.")
    st.markdown("---")

    if "df_current_full" not in st.session_state or st.session_state["df_current_full"] is None:
        st.warning("⚠️ Upload a dataset in Tab 1 first.")
        st.stop()

    df_full   = st.session_state["df_current_full"]
    total_rows = len(df_full)
    quarter    = total_rows // 4

    st.subheader("📊 Slice Mode")
    slice_option = st.radio(
        "Select a quarter of the dataset to analyze:",
        ("Top 25%", "25–50%", "50–75%", "Bottom 25%"),
        horizontal=True
    )
    slice_map = {
        "Top 25%":    df_full.iloc[:quarter],
        "25–50%":     df_full.iloc[quarter:quarter*2],
        "50–75%":     df_full.iloc[quarter*2:quarter*3],
        "Bottom 25%": df_full.iloc[quarter*3:]
    }
    df_sample = slice_map[slice_option]

    st.markdown("### 🔍 Dataset Summary")
    st.write(f"**Shape:** {df_sample.shape[0]} rows × {df_sample.shape[1]} columns")
    st.write("**Data Types:**")
    st.dataframe(df_sample.dtypes)

    missing_counts  = df_sample.isnull().sum()
    missing_percent = (missing_counts / len(df_sample)) * 100
    missing_df = pd.DataFrame({'Missing Values': missing_counts, 'Percent Missing': missing_percent}).round(2)
    st.write("**Missing Data:**")
    st.dataframe(missing_df[missing_df['Missing Values'] > 0])
    st.write(f"**Duplicate Rows:** {df_sample.duplicated().sum()}")

    st.markdown("### 📈 Numeric Distribution")
    numeric_cols = df_sample.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        stats_df = df_sample[numeric_cols].describe().T
        stats_df['skew']     = df_sample[numeric_cols].skew()
        stats_df['kurtosis'] = df_sample[numeric_cols].kurtosis()
        st.dataframe(stats_df.round(2))
    else:
        st.info("No numeric columns detected.")

    # WinBert narration of the summary
    if st.button("🤖 WinBert: Narrate This Summary"):
        with st.spinner("WinBert is reviewing the stats..."):
            summary_text = df_sample.describe().to_string()[:2000]
            missing_text = missing_df[missing_df['Missing Values'] > 0].to_string()
            system = "You are WinBert, a friendly data analyst. Narrate dataset statistics in plain English."
            user   = f"""
Narrate the following dataset statistics in 3–5 plain-English bullet points.
Flag anything unusual (skew, missing data, outliers). Be direct.

Statistical summary:
{summary_text}

Missing data:
{missing_text if missing_text.strip() else "None"}
"""
            st.markdown(ask_winbert(system, user))

    st.markdown("---")

    # Categorical distributions
    st.markdown("### Top Categorical Distributions")
    cat_cols = df_sample.select_dtypes(include='object').columns.tolist()
    if not cat_cols:
        st.info("No categorical columns found.")
    else:
        for i in range(0, len(cat_cols), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(cat_cols):
                    col_name = cat_cols[i + j]
                    with cols[j]:
                        count_df = df_sample[col_name].value_counts().head(5).reset_index()
                        count_df.columns = [col_name, "Count"]
                        fig = px.pie(count_df, names=col_name, values="Count", hole=0.3)
                        fig.update_layout(margin=dict(t=10, b=10), height=300)
                        st.markdown(f"**{col_name}**")
                        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Auto chart gallery
    st.markdown("### 🧪 Auto Chart Gallery")
    chart_type = st.radio("Chart style:", ["Bar (Counts)", "Line (Counts)"], horizontal=True)
    float_cols = df_sample.select_dtypes(include=["float"]).columns
    for col in float_cols:
        df_sample = df_sample.copy()
        df_sample[col] = df_sample[col].round().astype("Int64")
    numeric_cols = df_sample.select_dtypes(include="number").columns.tolist()

    if numeric_cols:
        for i in range(0, len(numeric_cols), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(numeric_cols):
                    col = numeric_cols[i + j]
                    with cols[j]:
                        try:
                            binned_col = pd.cut(df_sample[col], bins=10)
                            counts = binned_col.value_counts().sort_index()
                            clean_labels = [f"{int(iv.left)}–{int(iv.right)}" for iv in counts.index]
                            vc_df = pd.DataFrame({f"{col} (binned)": clean_labels, "Count": counts.values})
                            if chart_type == "Bar (Counts)":
                                fig = px.bar(vc_df, x=f"{col} (binned)", y="Count", title=f"{col}")
                            else:
                                fig = px.line(vc_df, x=f"{col} (binned)", y="Count", title=f"{col}")
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not chart {col}: {e}")

    st.markdown("---")

    # Categorical count explorer
    st.markdown("### Categorical Count Explorer")
    cat_cols = df_sample.select_dtypes(include='object').columns.tolist()
    if cat_cols:
        selected_cat = st.selectbox("Select a categorical column", cat_cols)
        top_n = st.slider("Top N values", 5, 30, 10)
        count_df = df_sample[selected_cat].value_counts().head(top_n).reset_index()
        count_df.columns = [selected_cat, "Count"]
        chart_style = st.radio("Chart style:", ["Bar", "Pie"], horizontal=True)
        st.dataframe(count_df)
        if chart_style == "Bar":
            st.plotly_chart(px.bar(count_df, x=selected_cat, y="Count"))
        else:
            st.plotly_chart(px.pie(count_df, names=selected_cat, values="Count"))


# ═══════════════════════════════════════════════════════════
# TAB 4 — FORECASTING
# ═══════════════════════════════════════════════════════════
with tab4:
    df_clean   = st.session_state.get("df_clean")
    df_raw     = st.session_state.get("df_raw")
    df_current = df_clean if df_clean is not None else df_raw

    if df_current is None:
        st.warning("⚠️ Upload a dataset in Tab 1 first.")
        st.stop()

    st.session_state["df_current"] = df_current
    st.info(f"Loaded: **{df_current.shape[0]}** rows × **{df_current.shape[1]}** columns")
    st.markdown("## 🔬 Forecasting & Advanced Analysis")

    # ── Linear Forecast ─────────────────────────────────────
    st.markdown("## 📈 Linear Forecast (Simple)")
    try:
        date_cols    = [col for col in df_current.columns if pd.api.types.is_datetime64_any_dtype(df_current[col])]
        numeric_cols = df_current.select_dtypes(include='number').columns.tolist()

        if not date_cols:
            st.warning("No datetime column found. Include a date column to enable forecasting.")
        else:
            date_col         = st.selectbox("Date column:", date_cols)
            target_col       = st.selectbox("Value to forecast:", numeric_cols)
            forecast_periods = st.slider("Months to forecast", 1, 12, 6)

            df_forecast = df_current[[date_col, target_col]].dropna().copy()
            df_forecast[date_col] = pd.to_datetime(df_forecast[date_col], errors='coerce')
            df_forecast = df_forecast.sort_values(date_col)
            df_forecast['ordinal_date'] = df_forecast[date_col].map(datetime.datetime.toordinal)

            model = LinearRegression()
            model.fit(df_forecast[['ordinal_date']], df_forecast[target_col])

            last_date    = df_forecast[date_col].max()
            future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, forecast_periods + 1)]
            predictions  = model.predict([[d.toordinal()] for d in future_dates])

            forecast_df = pd.DataFrame({date_col: future_dates, 'Forecast': predictions})
            full_df = pd.concat([
                df_forecast[[date_col, target_col]].rename(columns={target_col: "Actual"}),
                forecast_df.rename(columns={'Forecast': 'Actual'})
            ]).reset_index(drop=True)

            fig = px.line(full_df, x=date_col, y="Actual", title=f"Forecast for {target_col}", markers=True)
            fig.add_scatter(x=forecast_df[date_col], y=forecast_df['Forecast'], mode="lines+markers", name="Forecast")
            st.plotly_chart(fig, use_container_width=True)

            # WinBert narrates the forecast
            if st.button("🤖 WinBert: Interpret This Forecast"):
                with st.spinner("WinBert is analyzing the trend..."):
                    trend_dir   = "upward" if predictions[-1] > predictions[0] else "downward"
                    pct_change  = ((predictions[-1] - predictions[0]) / abs(predictions[0]) * 100) if predictions[0] != 0 else 0
                    system = "You are WinBert, a data analyst. Interpret forecasts clearly for business users."
                    user = f"""
Interpret this forecast for a business user:
- Column being forecast: {target_col}
- Forecast period: {forecast_periods} months
- Trend direction: {trend_dir}
- Projected change: {pct_change:.1f}%
- Starting forecast value: {predictions[0]:.2f}
- Ending forecast value: {predictions[-1]:.2f}

Give 2–3 plain-English sentences explaining what this means and any caveats about a simple linear model.
"""
                    st.markdown(ask_winbert(system, user))

    except Exception as e:
        st.error(f"Forecasting failed: {e}")

    # ── Prophet Forecast ────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🔮 Advanced Forecasting (Prophet)")
    with st.expander("ℹ️ Prophet Requirements"):
        st.markdown("""
- Time series data (monthly sales, daily counts, etc.)
- Minimum 12 time points
- Consistent date intervals
        """)
    try:
        date_cols    = [col for col in df_current.columns if pd.api.types.is_datetime64_any_dtype(df_current[col])]
        numeric_cols = df_current.select_dtypes(include='number').columns.tolist()

        if not date_cols:
            st.warning("No datetime column found.")
        else:
            date_col      = st.selectbox("Date column (Prophet):", date_cols, key="prophet_date")
            target_col    = st.selectbox("Value to forecast (Prophet):", numeric_cols, key="prophet_target")
            forecast_months = st.slider("Months to forecast (Prophet)", 1, 12, 6, key="prophet_months")

            df_prophet = df_current[[date_col, target_col]].dropna().copy()
            df_prophet.columns = ["ds", "y"]
            df_prophet["ds"] = pd.to_datetime(df_prophet["ds"], errors="coerce")
            df_prophet = df_prophet.dropna()

            m = Prophet()
            m.fit(df_prophet)
            future   = m.make_future_dataframe(periods=forecast_months * 30, freq='D')
            forecast = m.predict(future)

            st.write("📊 Forecast Table (tail):")
            st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_months * 30))
            st.pyplot(m.plot(forecast))
            st.pyplot(m.plot_components(forecast))

            # WinBert narration
            if st.button("🤖 WinBert: Interpret Prophet Forecast"):
                with st.spinner("WinBert is reading the forecast components..."):
                    tail = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30)
                    system = "You are WinBert, a data analyst. Explain Prophet forecasts clearly to business users."
                    user = f"""
Interpret this Prophet forecast for {target_col} over {forecast_months} months.
Tail of forecast:
{tail.to_string(index=False)}

Explain: trend direction, confidence interval width, and any caveats. 3–4 plain-English sentences max.
"""
                    st.markdown(ask_winbert(system, user))

    except Exception as e:
        st.error(f"Prophet forecasting failed: {e}")

    # ── Univariate Analysis ─────────────────────────────────
    st.markdown("---")
    st.markdown("## 📈 Univariate Analysis")
    num_cols = df_current.select_dtypes(include=np.number).columns.tolist()
    analysis_option = st.selectbox("Analysis type:", [
        "Single Column Analysis",
        "Histogram Grid",
        "Boxplot Grid"
    ])

    if analysis_option == "Single Column Analysis":
        selected_col = st.selectbox("Select column:", num_cols)
        if selected_col:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(px.histogram(df_current[selected_col].dropna(), nbins=30, title=f"Histogram: {selected_col}"), use_container_width=True)
            with col2:
                st.plotly_chart(px.box(df_current, y=selected_col, title=f"Boxplot: {selected_col}"), use_container_width=True)

    elif analysis_option == "Histogram Grid":
        if st.button("Generate Histograms"):
            cols = 3
            rows = -(-len(num_cols) // cols)
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
            axes = axes.flatten()
            for i, col in enumerate(num_cols):
                sns.histplot(data=df_current, x=col, ax=axes[i], kde=True)
                axes[i].set_title(col)
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            st.pyplot(fig)

    elif analysis_option == "Boxplot Grid":
        if st.button("Generate Boxplots"):
            cols = 3
            rows = -(-len(num_cols) // cols)
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
            axes = axes.flatten()
            for i, col in enumerate(num_cols):
                sns.boxplot(data=df_current, x=col, ax=axes[i])
                axes[i].set_title(col)
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            st.pyplot(fig)

    # ── Correlation Heatmap ─────────────────────────────────
    st.markdown("---")
    st.markdown("## 🔥 Correlation Heatmap")
    if st.button("Generate Heatmap"):
        corr = df_current.select_dtypes(include=np.number).corr()
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.heatmap(corr, annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral", ax=ax)
        st.pyplot(fig)

    # ── Bivariate Analysis ──────────────────────────────────
    st.markdown("---")
    st.markdown("## 🔁 Bivariate Analysis")
    num_cols = df_current.select_dtypes(include="number").columns.tolist()
    if len(num_cols) >= 2:
        biv_col1 = st.selectbox("First column:",  num_cols, key="biv_col1")
        biv_col2 = st.selectbox("Second column:", [c for c in num_cols if c != biv_col1], key="biv_col2")
        st.plotly_chart(px.scatter(df_current, x=biv_col1, y=biv_col2, trendline="ols", title=f"{biv_col1} vs {biv_col2}"), use_container_width=True)
        corr_val = df_current[biv_col1].corr(df_current[biv_col2])
        st.markdown(f"**Pearson r:** `{corr_val:.2f}`")

        if abs(corr_val) > 0.7:
            st.markdown(f"🧠 Strong {'positive' if corr_val > 0 else 'negative'} relationship between **{biv_col1}** and **{biv_col2}**.")
        elif abs(corr_val) > 0.3:
            st.markdown(f"🧠 Moderate correlation between **{biv_col1}** and **{biv_col2}**.")
        else:
            st.markdown(f"🧠 Little to no linear relationship between **{biv_col1}** and **{biv_col2}**.")
    else:
        st.warning("Need at least 2 numeric columns.")


# ═══════════════════════════════════════════════════════════
# TAB 5 — DATA SCIENCE TOOLS
# ═══════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 🔬 Data Science & ML Tools")

    df_clean   = st.session_state.get("df_clean")
    df_raw     = st.session_state.get("df_raw")
    df_sample  = st.session_state.get("df_sample")
    df_norm    = st.session_state.get("normalized_data")
    df_current = df_clean if df_clean is not None else df_raw
    st.session_state["df_current"] = df_current

    if df_sample is None:
        st.error("🚫 No dataset loaded. Upload in Tab 1.")
        st.stop()

    st.success("✅ Dataset loaded.")
    if df_norm is not None:
        st.info("🧪 Normalized dataset will be used for ML modeling.")
    else:
        st.warning("⚠️ No normalized data found. Normalize in Tab 1 for best results.")

    # ── Random Forest ───────────────────────────────────────
    st.markdown("## 🌲 Random Forest Regressor")
    data_for_modeling = df_norm if df_norm is not None else df_current

    if data_for_modeling is not None:
        numeric_cols = data_for_modeling.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) >= 2:
            target_col = st.selectbox("🎯 Target column:", numeric_cols)
            features   = [c for c in numeric_cols if c != target_col]
            st.write(f"Using **{len(features)}** features to predict **{target_col}**")

            st.sidebar.header("🛠️ Model Settings")
            n_estimators      = st.sidebar.slider("Trees (n_estimators)", 10, 500, 100, step=10)
            max_depth         = st.sidebar.slider("Max Depth", 1, 50, 10)
            min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 5)
            min_samples_leaf  = st.sidebar.slider("Min Samples Leaf",  1, 20, 2)

            if st.button("🌲 Run Random Forest"):
                X = data_for_modeling[features]
                y = data_for_modeling[target_col]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = RandomForestRegressor(
                    n_estimators=n_estimators, max_depth=max_depth,
                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                    max_features='sqrt', random_state=42, n_jobs=-1
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mae  = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2   = r2_score(y_test, y_pred)

                importances = model.feature_importances_
                feature_df  = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance", ascending=False)

                st.subheader("🔍 Feature Importances")
                st.dataframe(feature_df)
                st.plotly_chart(px.bar(feature_df, x="Feature", y="Importance", title="Feature Importance"))

                sample_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}).reset_index(drop=True)
                sample_df["Error"] = sample_df["Actual"] - sample_df["Predicted"]
                st.subheader("🎯 Actual vs. Predicted (sample)")
                st.dataframe(sample_df.head(10))
                st.plotly_chart(px.scatter(sample_df, x="Actual", y="Predicted", trendline="ols", title="Actual vs Predicted"))

                st.subheader("📈 Model Metrics")
                st.markdown(f"- **MAE:** `{mae:.2f}`")
                st.markdown(f"- **RMSE:** `{rmse:.2f}`")
                st.markdown(f"- **R²:** `{r2:.2f}`")

                with st.spinner("Running 5-fold cross-validation..."):
                    cv_score = cross_val_score(model, X, y, cv=5, scoring='r2').mean()
                st.markdown(f"- **CV R²:** `{cv_score:.2f}`")

                if r2 < 0.2:
                    st.warning("⚠️ Low R² — model may not be capturing the signal well.")
                elif r2 > 0.9 and (r2 - cv_score) > 0.1:
                    st.warning("⚠️ Possible overfitting — CV R² is much lower than training R².")

                # WinBert explains the results
                if st.button("🤖 WinBert: Explain These Results"):
                    with st.spinner("WinBert is interpreting the model..."):
                        top_features = feature_df.head(3)["Feature"].tolist()
                        system = "You are WinBert, a data analyst. Explain ML model results to non-technical business users."
                        user = f"""
Explain these Random Forest results to a business owner in plain English:
- Target variable: {target_col}
- R² score: {r2:.2f}
- Cross-validated R²: {cv_score:.2f}
- MAE: {mae:.2f}
- Top 3 most important features: {', '.join(top_features)}

Explain what R² means in plain terms, what the top features tell us, and whether this model is trustworthy.
Keep it under 5 sentences.
"""
                        st.markdown(ask_winbert(system, user))

        else:
            st.warning("Need at least 2 numeric columns.")

    # ── Statistical Summary ─────────────────────────────────
    st.markdown("---")
    st.markdown("## 🧮 Core Statistical Summary")
    if df_current is not None:
        numeric_cols = df_current.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            desc_df = df_current[numeric_cols].describe().T
            desc_df["iqr"]      = df_current[numeric_cols].quantile(0.75) - df_current[numeric_cols].quantile(0.25)
            desc_df["skew"]     = df_current[numeric_cols].skew()
            desc_df["kurtosis"] = df_current[numeric_cols].kurtosis()
            desc_df = desc_df.rename(columns={"mean": "Mean", "50%": "Median", "std": "Std Dev",
                                               "min": "Min", "max": "Max", "iqr": "IQR",
                                               "skew": "Skew", "kurtosis": "Kurtosis"})
            st.dataframe(desc_df[["Mean", "Median", "Std Dev", "Min", "Max", "IQR", "Skew", "Kurtosis"]].round(2))

            if st.checkbox("🤖 WinBert: Explain the Stats"):
                for col in numeric_cols[:5]:
                    sk = df_current[col].skew()
                    note = "right-skewed (tail to the right)" if sk > 1 else "left-skewed (tail to the left)" if sk < -1 else "fairly symmetrical"
                    st.markdown(f"**{col}** is *{note}* (skewness = `{sk:.2f}`)")

    # ── Hypothesis Testing ──────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Hypothesis Testing")
    test_type = st.radio("Test type:", ["One-sample t-test", "Two-sample t-test", "Z-test", "Chi-square test"], horizontal=True)

    if test_type == "One-sample t-test":
        column  = st.selectbox("Numeric column:", df_current.select_dtypes(include='number').columns)
        popmean = st.number_input("Population mean to test against:", value=0.0)
        if len(df_current[column].dropna()) > 1:
            t_stat, p_val = stats.ttest_1samp(df_current[column].dropna(), popmean)
            st.write(f"T-statistic: `{t_stat:.4f}`  |  P-value: `{p_val:.4f}`")
            if p_val < 0.05:
                st.markdown(f"🧠 Significant difference from mean of `{popmean}` (p = {p_val:.4f}).")
            else:
                st.markdown(f"🧠 No significant difference from mean of `{popmean}` (p = {p_val:.4f}).")

    elif test_type == "Two-sample t-test":
        num_col = st.selectbox("Numeric column:", df_current.select_dtypes(include='number').columns)
        cat_col = st.selectbox("Binary categorical column:", df_current.select_dtypes(include='object').columns)
        unique_vals = df_current[cat_col].dropna().unique()
        if len(unique_vals) == 2:
            g1 = df_current[df_current[cat_col] == unique_vals[0]][num_col].dropna()
            g2 = df_current[df_current[cat_col] == unique_vals[1]][num_col].dropna()
            t_stat, p_val = stats.ttest_ind(g1, g2)
            st.write(f"T-statistic: `{t_stat:.4f}`  |  P-value: `{p_val:.4f}`")
            if p_val < 0.05:
                st.markdown(f"🧠 Significant difference in **{num_col}** between groups.")
            else:
                st.markdown(f"🧠 No significant difference in **{num_col}** between groups.")
        else:
            st.warning("Select a column with exactly 2 unique values.")

    elif test_type == "Z-test":
        column   = st.selectbox("Numeric column:", df_current.select_dtypes(include='number').columns, key="z_col")
        pop_mean = st.number_input("Population Mean:", value=0.0)
        pop_std  = st.number_input("Population Std Dev:", value=1.0)
        sample   = df_current[column].dropna()
        if len(sample) > 1:
            z_stat = (sample.mean() - pop_mean) / (pop_std / np.sqrt(len(sample)))
            p_val  = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            st.write(f"Z-statistic: `{z_stat:.4f}`  |  P-value: `{p_val:.4f}`")
            if p_val < 0.05:
                st.markdown(f"🧠 Significant difference from population mean of `{pop_mean}`.")
            else:
                st.markdown(f"🧠 No significant difference from population mean of `{pop_mean}`.")

    elif test_type == "Chi-square test":
        col1 = st.selectbox("First categorical column:",  df_current.select_dtypes(include='object').columns, key="chi1")
        col2 = st.selectbox("Second categorical column:", df_current.select_dtypes(include='object').columns, key="chi2")
        if col1 != col2:
            contingency = pd.crosstab(df_current[col1], df_current[col2])
            chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
            st.write(f"Chi² = `{chi2:.4f}` | p = `{p_val:.4f}` | df = `{dof}`")
            if p_val < 0.05:
                st.markdown(f"🧠 Significant relationship between **{col1}** and **{col2}**.")
            else:
                st.markdown(f"🧠 No significant association between **{col1}** and **{col2}**.")
        else:
            st.warning("Select two different columns.")

    # ── Auto Statistical Insights ───────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Auto Statistical Insights")
    if st.checkbox("Run Statistical Scan"):
        results = run_auto_statistical_insights(df_current)
        if results:
            for r in results:
                st.markdown(r)
        else:
            st.info("No statistically significant findings detected.")

    # ── SPC Charts ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📉 SPC Charts & Process Monitoring")
    numeric_cols = df_current.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        selected_col   = st.selectbox("Column to analyze:", numeric_cols)
        subgroup_size  = st.slider("Subgroup size", 3, 10, 5)
        show_insight   = st.checkbox("💡 Show WinBert Insight")
        if st.button("Generate SPC Chart"):
            xbar_r_chart(df_current, selected_col, subgroup_size, show_insight)
    else:
        st.warning("No numeric columns available.")

    with st.expander("📊 SPC by Group"):
        cat_cols_spc = df_current.select_dtypes(include='object').columns.tolist()
        num_cols_spc = df_current.select_dtypes(include='number').columns.tolist()
        if cat_cols_spc and num_cols_spc:
            grp_col    = st.selectbox("Group by:", cat_cols_spc)
            metric_col = st.selectbox("SPC metric:", num_cols_spc)
            sub_size   = st.slider("Subgroup size", 3, 10, 5, key="grp_sub")
            grp_insight = st.checkbox("💡 WinBert insight per chart", key="grp_ins")
            if st.button("📈 Generate Group SPC Charts"):
                for grp in sorted(df_current[grp_col].dropna().unique()):
                    subset = df_current[df_current[grp_col] == grp][metric_col].dropna().reset_index(drop=True)
                    if len(subset) < sub_size * 2:
                        st.warning(f"Not enough data for group `{grp}`.")
                        continue
                    st.subheader(f"🔹 {grp_col}: `{grp}`")
                    xbar_r_chart(subset.to_frame(), metric_col, sub_size, grp_insight)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
year = datetime.datetime.now().year
st.markdown(f"""
<div style='text-align: center; font-size: 0.9em; color: gray; margin-top: 50px;'>
  <hr>
  <p>Built with ❤️ using Streamlit · Local Llama 3 (Ollama) · Pandas · Plotly</p>
  <p>📬 <a href="mailto:pocketanalyst.help@gmail.com">pocketanalyst.help@gmail.com</a></p>
  <p>© {year} WinBert AI | Pocket Analyst · DKR Ventures LLC</p>
</div>
""", unsafe_allow_html=True)
