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
        
     # --- Trial-limited Predictive Modeling ---
    if "predict_use_count" not in st.session_state:
        st.session_state.predict_use_count = 0

    if st.session_state.predict_use_count < 3:
        with st.expander("🔮 Predictive Modeling (Beta)"):
            st.markdown("Use linear regression to forecast outcomes based on your data. You have "
                        f"{3 - st.session_state.predict_use_count} predictions left today.")
            numeric_cols = df_sample.select_dtypes(include=np.number).columns.tolist()

            target_col = st.selectbox("Select a column to predict (target):", options=numeric_cols)
            feature_cols = st.multiselect("Select feature columns to use for prediction:", options=[col for col in numeric_cols if col != target_col])

            if target_col and feature_cols:
                new_data = []
                st.markdown("Enter values for prediction:")
                for col in feature_cols:
                    val = st.number_input(f"{col}", step=1.0)
                    new_data.append(val)

                if st.button("Run Prediction"):
                    try:
                        X = df_sample[feature_cols]
                        y = df_sample[target_col]

                        model = LinearRegression()
                        model.fit(X, y)

                        prediction = model.predict([new_data])[0]
                        st.success(f"Estimated {target_col}: {prediction:,.2f}")
                        st.session_state.predict_use_count += 1
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
    else:
        with st.expander("🔒 Predictive Modeling (Pro Only)"):
            st.warning("You've reached your free trial limit for predictions today. Upgrade to Pocket Analyst Pro to unlock unlimited forecasting.")
            st.button("🔓 Unlock Predictive Tools")


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

    # --- Chat Section ---
    user_question = st.text_input("Ask a question about your data:")

    # Session usage tracking (limit free users to 5 questions/day)
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
        st.session_state.query_reset_time = datetime.datetime.now()

    if datetime.datetime.now() - st.session_state.query_reset_time > datetime.timedelta(days=1):
        st.session_state.query_count = 0
        st.session_state.query_reset_time = datetime.datetime.now()

    if user_question:
        if st.session_state.query_count >= 5:
            st.warning("You've reached your free question limit for today. Please upgrade to unlock more features.")
        else:
            st.session_state.query_count += 1
            csv_snippet = df_sample.to_csv(index=False)

            prompt = f"""
            You are an expert data analyst. Based on the following CSV data, answer the user's question clearly and briefly. Do not include Python code in your response.

            Data:
            {csv_snippet[:4000]}

            Question: {user_question}
            """

            try:
                response = openai.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful data analyst."},
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = response.choices[0].message.content

                st.subheader("Bot's Answer")
                with st.expander("AI Response", expanded=True):
                    st.write(answer)

                # Dynamic chart rendering
                chart_type, chart_cols = detect_chart_type_and_columns(user_question, df_sample)

                if chart_type == "bar" and chart_cols and chart_cols in df_sample.columns:
                    fig = px.bar(df_sample, x=chart_cols)
                    st.plotly_chart(fig)

                elif chart_type == "line" and all(chart_cols):
                    fig = px.line(df_sample, x=chart_cols[0], y=chart_cols[1])
                    st.plotly_chart(fig)

                elif chart_type == "scatter" and all(chart_cols):
                    fig = px.scatter(df_sample, x=chart_cols[0], y=chart_cols[1], color=chart_cols[2])
                    st.plotly_chart(fig)

            except Exception as e:
                st.error(f"API Error: {e}")
else:
    st.info("Please upload a file to get started.")


    # Dynamic chart rendering from natural question
    chart_type, chart_cols = detect_chart_type_and_columns(user_question, df_sample)

        if chart_type == "bar" and chart_cols and chart_cols in df_sample.columns:
        fig = px.bar(df_sample, x=chart_cols)
        st.plotly_chart(fig)

        elif chart_type == "line" and chart_cols and all(chart_cols):
        fig = px.line(df_sample, x=chart_cols[0], y=chart_cols[1])
        st.plotly_chart(fig)

        elif chart_type == "scatter" and chart_cols and all(chart_cols):
        fig = px.scatter(df_sample, x=chart_cols[0], y=chart_cols[1], color=chart_cols[2])
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"API Error: {e}")

            
# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ using Streamlit + OpenAI + Pandas + Plotly")
st.markdown("📬 Need help? Contact us at [pocketanalyst.help@gmail.com](mailto:pocketanalyst.help@gmail.com)")
