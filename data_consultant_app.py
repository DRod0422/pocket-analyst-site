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
    page_title=" AI Pocket Analyst",
    layout="wide",
    page_icon="logo.png"
)
#st.image("logo.png", width=160)
st.title("🤖 AI Pocket Analyst")
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
        # Get available sheet names
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        selected_sheet = st.selectbox("Select a sheet to load", sheet_names)
        
        # Load only the selected sheet
        df = pd.read_excel(xls, sheet_name=selected_sheet)
        
    if "last_uploaded_name" not in st.session_state or st.session_state.last_uploaded_name != uploaded_file.name:
        st.session_state.last_uploaded_name = uploaded_file.name
        st.session_state.ai_ran_once = False  # ✅ Reset ONLY on new file
        
    st.subheader("Preview of Your Data")
    st.dataframe(df.head(100))

    if len(df) > 5000:
        st.warning(f"Large dataset detected ({len(df)} rows). Sampling 1000 rows for faster performance.")
        df_sample = df.sample(n=1000, random_state=42)
    else:
        df_sample = df
        
    # # ✅ Reset AI trigger
    # if uploaded_file and "ai_ran_once" not in st.session_state: 
    #     st.session_state.ai_ran_once = False
    
    # --- Normalize Data ---
    with st.expander("🧹 Data Normalization & Encoding", expanded=False):
        st.markdown("""
        Prepare your dataset for analysis by normalizing numeric values and encoding categories.
        
        - **One-hot encoding**: Converts categories into numeric flags.
        - **Normalization**: Scales numbers between 0 and 1.
        - Optionally: Drop columns manually before running ML models.
        """)
        
        drop_columns = st.multiselect("Optional: Drop Columns Before Processing", df.columns.tolist())

        # Select scaler
        scaler_choice = st.selectbox(
            "Choose a normalization method:",
            ("MinMaxScaler", "StandardScaler", "RobustScaler")
        )

        skip_scaling = st.checkbox("⚠️ Skip normalization (my data is already scaled)")
        
        normalize_data = st.button("⚙️ Normalize & Encode Dataset")

        if normalize_data:
            from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    
            # Prepare base dataframe
            df_encoded = df.drop(columns=drop_columns) if drop_columns else df.copy()
    
            # One-hot encode categoricals
            df_encoded = pd.get_dummies(df_encoded, drop_first=True)
    
            # Normalize numeric columns if not skipped
            numeric_cols = df_encoded.select_dtypes(include=["int64", "float64"]).columns
    
            if not skip_scaling and len(numeric_cols) > 0:
                if scaler_choice == "MinMaxScaler":
                    scaler = MinMaxScaler()
                elif scaler_choice == "StandardScaler":
                    scaler = StandardScaler()
                else:
                    scaler = RobustScaler()
    
                df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
        
            # Store it in session state
            st.session_state["normalized_data"] = df_encoded
        
            st.success("✅ Dataset normalized and one-hot encoded!")
            st.dataframe(df_encoded.head())

            
    # --- Quick AI Insights block ---
    if uploaded_file:
        if "ai_ran_once" not in st.session_state:
            st.session_state.ai_ran_once = False
    
        # Show disabled state if already run
        if st.session_state.ai_ran_once:
            st.success("✅ AI Insights already generated for this session.")
        else:
            if st.button("🧠 Generate AI Insights"):
                with st.expander("✨ AI Quick Insights", expanded=True):
                    try:
                        st.markdown("Here's what I noticed in your data:")
    
                        csv_snippet = df_sample.to_csv(index=False)[:4000]
                        insight_prompt = f"""
                        You are a data analyst. Read the data below and write 3 short, plain-English insights.
                        Avoid technical jargon. Pretend you're talking to a small business owner.
    
                        Data sample:
                        {csv_snippet}
                        """
    
                        response = openai.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "You are a helpful data analyst."},
                                {"role": "user", "content": insight_prompt}
                            ]
                        )
                        ai_insights = response.choices[0].message.content
    
                        st.markdown(ai_insights)
                        st.session_state.ai_ran_once = True  # 🔒 Lock it
    
                    except Exception as e:
                        st.warning(f"Could not generate AI insights: {e}")


    # --- Go-By Suggestions ---
    with st.expander("💡 Try asking about your data:"):
        st.markdown("""
        - What are the key trends in this dataset?
        - What is the average sale price?
        - How many items were sold per region?
        - Which month had the highest revenue?
        - What insights can you provide from this data?
        """)
        
    # --- Chat Section ---
    user_question = st.text_input("Ask a question about your data:")

    # --- Light Sampling for Large Files ---
        
    if len(df) > 5000:
        st.warning(f"Large dataset detected ({len(df)} rows). Sampling 1000 rows for efficiency.")
        df_sample = df.sample(n=1000, random_state=42)
    else:
        df_sample = df
        
    

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
            csv_snippet = df_sample.head(10).to_string(index=False)

            row_count, col_count = df_sample.shape
            
            prompt = f"""
            You are an expert data analyst. The dataset has {row_count} rows and {col_count} columns.
            Below is a preview of the first 10 rows. Use it to understand the structure and help answer the user's question.

            Sample Data:
            {csv_snippet}
            

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
                
    # --- Divider --
    st.markdown("---")            

    # --- Smart Auto Insights ---
    with st.expander("Smart Auto Insights", expanded=True):
        st.markdown("Get an instant overview of your dataset without lifting a finger. This section auto-generates summaries, stats, and visuals for quick insight.")

        st.subheader("🔍 Dataset Summary")
        st.write(f"**Shape:** {df_sample.shape[0]} rows × {df_sample.shape[1]} columns")
        st.write("**Data Types:**")
        st.dataframe(df_sample.dtypes)

        missing_counts = df_sample.isnull().sum()
        missing_percent = (missing_counts / len(df_sample)) * 100
        missing_df = pd.DataFrame({
        'Missing Values': missing_counts,
        'Percent Missing': missing_percent
        }).round(2)
        st.write("**Missing Data Overview:**")
        st.dataframe(missing_df[missing_df['Missing Values'] > 0])

        dup_count = df_sample.duplicated().sum()
        st.write(f"**Duplicate Rows:** {dup_count}")

        st.subheader("📈 Quick Distribution Check (Numeric Columns)")
        numeric_cols = df_sample.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            stats_df = df_sample[numeric_cols].describe().T
            stats_df['skew'] = df_sample[numeric_cols].skew()
            stats_df['kurtosis'] = df_sample[numeric_cols].kurtosis()
            st.dataframe(stats_df.round(2))
        else:
            st.info("No numeric columns detected.")

        st.subheader("Top Categorical Distributions")
        cat_cols = df_sample.select_dtypes(include='object').columns.tolist()
        for col in cat_cols[:3]:  # Show only top 3 for brevity
            st.markdown(f"**{col}** - Top Categories")
            st.dataframe(df_sample[col].value_counts().head(5))

        st.subheader("🧪 Auto-Generated Chart Gallery")
        chart_type = st.radio("Chart style:", ["Bar (Counts)", "Line (Counts)"], horizontal=True)
        
        # Convert float columns to rounded integers (safe for counting)
        float_cols = df_sample.select_dtypes(include=["float"]).columns
        for col in float_cols:
            df_sample.loc[:, col] = df_sample[col].round().astype("Int64")
        
        # Recompute numeric columns after transformation
        numeric_cols = df_sample.select_dtypes(include="number").columns.tolist()

        if numeric_cols:
            st.markdown("Quick glance at value distributions:")

            for col in numeric_cols:
                st.markdown(f"**{col}**")
                try:
                    # Bin the numeric column into discrete intervals before counting
                    binned_col = pd.cut(df_sample[col], bins=10)  # You can tweak bin count if needed
                    counts = binned_col.value_counts().sort_index()
                    vc_df = pd.DataFrame({f"{col} (binned)": counts.index.astype(str), "Count": counts.values})

                    if chart_type == "Bar (Counts)":
                        fig = px.bar(vc_df, x=f"{col} (binned)", y="Count", title=f"{col} - Bar Chart (Binned)")
                    elif chart_type == "Line (Counts)":
                        fig = px.line(vc_df, x=f"{col} (binned)", y="Count", title=f"{col} - Line Chart (Binned)")
                    else:
                        st.warning("Chart type not recognized.")
                        
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate chart for {col}: {e}")
        else:
            st.info("No numeric columns found.")

                
    # --- Guidance for ML Tools --
    st.markdown("---")
    st.markdown("## 🔬 Modeling & Advanced Analysis")
    st.info(
        """
        The following tools include **univariate analysis**, **variable relationships**, and **predictive forecasting analysis**.
        These features require well-structured data — not all datasets are suitable.

        If your data is missing key variables, has poor formatting, or doesn't represent measurable relationships,
        these models may return inaccurate or meaningless results.

        ➤ Use these tools only when your data is appropriate for modeling.
        """
    )  

    # --- Predictive Forecasting (Simple Time Series) ---
    with st.expander("📈 Forecast Future Values (Beta)", expanded=False):
        try:
            date_cols = [col for col in df_sample.columns if pd.api.types.is_datetime64_any_dtype(df_sample[col])]
            numeric_cols = df_sample.select_dtypes(include='number').columns.tolist()
    
            if not date_cols:
                st.warning("No datetime column found. Please include a date column to enable forecasting.")
            else:
                date_col = st.selectbox("Select the date column:", date_cols)
                target_col = st.selectbox("Select the value to forecast:", numeric_cols)
    
                # User input for forecast horizon
                forecast_periods = st.slider("Months to forecast", min_value=1, max_value=12, value=6)
    
                # Prepare data
                df_forecast = df_sample[[date_col, target_col]].dropna().copy()
                df_forecast[date_col] = pd.to_datetime(df_forecast[date_col], errors='coerce')
                
                # Detect if any date has only a year (e.g., 2023) by checking if all dates are Jan 1
                if (df_forecast[date_col].dt.month == 1).all() and (df_forecast[date_col].dt.day == 1).all():
                    df_forecast[date_col] = df_forecast[date_col].dt.to_period("Y").dt.to_timestamp()
                else:
                    df_forecast[date_col] = df_forecast[date_col].dt.normalize()
                df_forecast = df_forecast.sort_values(date_col)

                # Convert dates to ordinal for regression
                df_forecast['ordinal_date'] = df_forecast[date_col].map(datetime.datetime.toordinal)
                X = df_forecast[['ordinal_date']]
                y = df_forecast[target_col]
    
                model = LinearRegression()
                model.fit(X, y)
    
                # Forecast future dates
                last_date = df_forecast[date_col].max()
                future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, forecast_periods + 1)]
                future_ordinal = [[d.toordinal()] for d in future_dates]
                predictions = model.predict(future_ordinal)
    
                # Create forecast DataFrame
                forecast_df = pd.DataFrame({
                    date_col: future_dates,
                    'Forecast': predictions
                })
                forecast_df[date_col] = pd.to_datetime(forecast_df[date_col]).dt.normalize()

                # Rename forecast column first
                forecast_df_renamed = forecast_df.rename(columns={'Forecast': "Actual"}, inplace=True)
    
                # Combine past + forecast
                full_df = pd.concat([
                    df_forecast[[date_col, target_col]].rename(columns={target_col: "Actual"}),
                    forecast_df_renamed
                ]).reset_index(drop=True)

                # 🔍 Detect if the dates are yearly or monthly
                if (df_forecast[date_col].dt.month == 1).all() and (df_forecast[date_col].dt.day == 1).all():
                    forecast_title = f"📅 Yearly Forecast for {target_col}"
                else:
                    forecast_title = f"📆 Monthly Forecast for {target_col}"

                # Plot
                fig = px.line(full_df, x=date_col, y="Actual", title=forecast_title, markers=True)
                fig.add_scatter(x=forecast_df[date_col], y=forecast_df["Actual"], mode="lines+markers", name="Forecast")
    
                st.plotly_chart(fig, use_container_width=True)
    
        except Exception as e:
            st.error(f"Forecasting failed: {e}")

    
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
                st.write(col_data.describe().T)
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

    # --- Variable Relationships ---
    with st.expander("🔍 Explore Variable Relationships"):
        st.markdown("### 🔥 Correlation Heatmap")
        if st.button("Generate Correlation Heatmap"):
            corr = df_sample.select_dtypes(include=np.number).corr()
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.heatmap(corr, annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral", ax=ax)
            st.pyplot(fig)

    # --- Divider ---
    st.markdown("---")
    st.markdown("## Data Science & Machine Learning Modeling")
    st.info("This section includes advanced machine learning tools for data scientists and experienced analysts.")
    
    # --- Advanced Data Scientist Tools (Expandable Section) ---
    with st.expander("🔬 Data Scientist Tools (Pro Preview) *Beta* ", expanded=False):
        
        # 👇 This handles switching between original or normalized data
        data_for_modeling = st.session_state.get("normalized_data", df)
        
        if uploaded_file is not None:
            try:
                numeric_cols = data_for_modeling.select_dtypes(include="number").columns.tolist()
    
                if len(numeric_cols) < 2:
                    st.warning("Not enough numeric columns to run advanced models.")
                else:
                    target_col = st.selectbox("🎯 Select a target column", numeric_cols)
    
                    features = [col for col in numeric_cols if col != target_col]
    
                    st.write(f"📊 Using {len(features)} features to predict **{target_col}**")

                    if st.checkbox("📘 What This Model Does"):
                        st.markdown("""
                        The **Random Forest** model is a machine learning algorithm that builds multiple decision trees and averages their results for better accuracy.
                    
                        - **Feature Importance** shows which variables influence the prediction the most.
                        - **Actual vs Predicted** helps you evaluate how well the model is performing.
                        - **Performance Metrics (MAE, RMSE, R²)** indicate how reliable the model is.
                    
                        **Example Use Cases:**
                        - Predict loan duration based on age, income, and credit score.
                        - Discover which features most impact an outcome.
                        - Explore "what-if" scenarios to understand data sensitivity.
                    
                        Use this model to gain deeper, data-backed insight into trends and patterns hidden in your dataset.
                        """)

                    if st.button("🌲 Run Random Forest Model"):
                        from sklearn.ensemble import RandomForestRegressor
                        from sklearn.model_selection import train_test_split
    
                        X = data_for_modeling[features]
                        y = data_for_modeling[target_col]

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
                        model = RandomForestRegressor(random_state=42)
                        model.fit(X_train, y_train)
    
                        importances = model.feature_importances_
                        feature_df = pd.DataFrame({
                            "Feature": features,
                            "Importance": importances
                        }).sort_values(by="Importance", ascending=False)
    
                        st.subheader("🔍 Feature Importances")
                        st.dataframe(feature_df)
    
                        import plotly.express as px
                        fig = px.bar(feature_df, x="Feature", y="Importance", title="Feature Importance (Random Forest)")
                        st.plotly_chart(fig)

                        # Show prediction samples
                        st.subheader("Prediction Samples (Actual vs. Predicted)")
                        sample_df = X_test.copy()
                        sample_df["Actual"] = y_test.values
                        sample_df["Predicted"] = model.predict(X_test)

                        st.caption("Compare model predictions to actual values for a sample of test data.")
                        st.dataframe(sample_df.head(10), use_container_width=True)

                        # Evaluate model performance
                        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                        import numpy as np
                        
                        # Generate predictions on test set
                        y_pred = model.predict(X_test)
                        
                        # Create a DataFrame of actual vs. predicted
                        sample_df = pd.DataFrame({
                            "Actual": y_test.values,
                            "Predicted": y_pred
                        }).reset_index(drop=True)
                        
                        st.subheader("🎯 Prediction Samples (Actual vs. Predicted)")
                        st.dataframe(sample_df.head(10))
                        
                        # Show performance metrics
                        mae = mean_absolute_error(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        r2 = r2_score(y_test, y_pred)
                        
                        st.subheader("📈 Model Performance Metrics")
                        st.markdown(f"- **MAE (Mean Absolute Error):** `{mae:.2f}`")
                        st.markdown(f"- **RMSE (Root Mean Squared Error):** `{rmse:.2f}`")
                        st.markdown(f"- **R² Score:** `{r2:.2f}`")

    
            except Exception as e:
                st.error(f"❌ Error running advanced analysis: {e}")


    
else:
    st.info("Please upload a file to get started.")

            
# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ using Streamlit + OpenAI + Pandas + Plotly")
st.markdown("📬 Need help? Contact us at [pocketanalyst.help@gmail.com](mailto:pocketanalyst.help@gmail.com)")
