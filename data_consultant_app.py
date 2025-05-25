# Chatbot Data Consultant Blueprint - Streamlit App

import streamlit as st
import pandas as pd
import openai
import plotly.express as px
import io
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import skew, kurtosis

# --- Config Section ---
openai.api_key = st.secrets["OPENAI_API_KEY"]
GPT_MODEL = "gpt-4o"  # GPT-4.1 Mini/Nano

# --- Page Config ---
st.set_page_config(
    page_title="🤖 Pocket Analyst",
    layout="wide",
    page_icon="logo.png"
)
st.image("logo.png", width=160)
st.title("Pocket Analyst")
st.caption("Upload your file. Ask questions. Predict outcomes. Get insights.")

# --- Chart Type Detector ---
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

# --- File Upload Section ---
uploaded_file = st.file_uploader("Upload a CSV or XLSX file", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Preview of Your Data")
    st.dataframe(df.head(100))

    # --- Go-By Suggestions ---
    with st.expander("💡 Try asking about your data:"):
        st.markdown("""
        - What are the key trends in this dataset?
        - What is the average sale price?
        - How many items were sold per region?
        - Which month had the highest revenue?
        - Predict the sale price for a 3-bedroom, 2-bath house
        - Show a bar chart of sales by category
        - Line chart of total revenue over time
        - What insights can you provide from this data?
        """)

    # --- Light Sampling for Large Files ---
    if len(df) > 5000:
        st.warning(f"Large dataset detected ({len(df)} rows). Sampling 1000 rows for efficiency.")
        df_sample = df.sample(n=1000, random_state=42)
    else:
        df_sample = df

    # --- Optional Chart Builder ---
    with st.expander("🛠️ Create a Custom Chart (optional)"):
        chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter"])
        x_col = st.selectbox("Select X-axis column", options=df_sample.columns)
        y_col = st.selectbox("Select Y-axis column", options=df_sample.columns)
        if st.button("Generate Custom Chart"):
            try:
                if chart_type == "Bar":
                    fig = px.bar(df_sample, x=x_col, y=y_col)
                elif chart_type == "Line":
                    fig = px.line(df_sample, x=x_col, y=y_col)
                elif chart_type == "Scatter":
                    fig = px.scatter(df_sample, x=x_col, y=y_col)
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Chart creation failed: {e}")

    # --- Univariate Analysis ---
    with st.expander("📈 Univariate Analysis"):
        num_cols = df_sample.select_dtypes(include=np.number).columns.tolist()
        analysis_option = st.selectbox("Select a univariate analysis option:", [
            "Single Column Analysis",
            "Histogram Grid for All Numeric Columns",
            "Boxplot Grid for All Numeric Columns"
        ])

        if analysis_option == "Single Column Analysis":
            selected_col = st.selectbox("Select a numeric column for analysis:", options=num_cols)
            if selected_col:
                col_data = df_sample[selected_col].dropna()
                st.write(f"**Summary Statistics for {selected_col}:**")
                st.write(col_data.describe())
                st.write(f"**Skewness:** {skew(col_data):.3f}")
                st.write(f"**Kurtosis:** {kurtosis(col_data):.3f}")
                st.plotly_chart(px.histogram(col_data, nbins=30, title=f"Histogram of {selected_col}"))
                st.plotly_chart(px.box(df_sample, y=selected_col, title=f"Boxplot of {selected_col}"))

        elif analysis_option == "Histogram Grid for All Numeric Columns":
            if st.button("Generate Histograms"):
                num_features = len(num_cols)
                cols = 3
                rows = -(-num_features // cols)
                fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
                axes = axes.flatten()
                for i, col in enumerate(num_cols):
                    sns.histplot(data=df_sample, x=col, ax=axes[i], kde=True)
                    axes[i].set_title(f"Histogram of {col}")
                for j in range(i + 1, len(axes)):
                    fig.delaxes(axes[j])
                st.pyplot(fig)

        elif analysis_option == "Boxplot Grid for All Numeric Columns":
            if st.button("Generate Boxplots"):
                num_features = len(num_cols)
                cols = 3
                rows = -(-num_features // cols)
                fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
                axes = axes.flatten()
                for i, col in enumerate(num_cols):
                    sns.boxplot(data=df_sample, x=col, ax=axes[i])
                    axes[i].set_title(f"Boxplot of {col}")
                for j in range(i + 1, len(axes)):
                    fig.delaxes(axes[j])
                st.pyplot(fig)

    # --- Machine Learning Insights ---
    with st.expander("🧠 Machine Learning Insights"):
        st.markdown("### 🔥 Correlation Heatmap")
        if st.button("Generate Correlation Heatmap"):
            corr = df_sample.select_dtypes(include=np.number).corr()
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.heatmap(corr, annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral", ax=ax)
            st.pyplot(fig)
