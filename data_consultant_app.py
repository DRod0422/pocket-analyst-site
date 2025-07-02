# Chatbot Data Consultant Blueprint - Streamlit App

import streamlit as st
import pandas as pd
import numpy as np
import io
import datetime
import base64
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from utils import clean_and_format_data

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet

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

# --- Tabs Layout ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Upload & Clean Data", 
    "🤖 AI Assistant",                         #📊 Quick Analysis
    "📊 Quick Analysis", 
    "📈 Forecasting", 
    "📐 Data Science & Statistical Tools"
])

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
with tab1:
    uploaded_file = st.file_uploader("Upload a CSV or XLSX file", type=["csv", "xlsx"])
    
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        else:
            # Get available sheet names
            xls = pd.ExcelFile(uploaded_file)
            sheet_names = xls.sheet_names
            selected_sheet = st.selectbox("Select a sheet to load", sheet_names)
            df_raw = pd.read_excel(xls, sheet_name=selected_sheet)
    
        # ✅ Only clean the data if it's new or changed
        if "last_uploaded_name" not in st.session_state or st.session_state.last_uploaded_name != uploaded_file.name:
            st.session_state.last_uploaded_name = uploaded_file.name
            st.session_state.ai_ran_once = False  # Reset AI insights on new file
        
            # Ask user if they want cleaning
            use_cleaning = st.checkbox("🧼 Auto-clean uploaded data?", value=True)
            st.session_state["use_cleaning"] = use_cleaning
        
            if use_cleaning:
                from utils import clean_and_format_data
                df_clean, clean_log = clean_and_format_data(df_raw, log=True)
                st.session_state["df_clean"] = df_clean
        
                st.success("✅ File cleaned and loaded.")
                for entry in clean_log:
                    st.markdown(f"🧼 {entry}")
            else:
                st.session_state["df_clean"] = None
        else:
            use_cleaning = st.session_state.get("use_cleaning", True)
            df_clean = st.session_state.get("df_clean")
        
        # Define df_sample from correct source
        if use_cleaning and df_clean is not None:
            working_df = df_clean
        else:
            working_df = df_raw
        
        
        # Select which dataset to use for downstream tasks
        apply_cleaning = st.checkbox("🧼 Apply Auto-Cleaning to Dataset", value=True)
        # Decide which dataset to use
        df_current = st.session_state["df_clean"] if apply_cleaning and "df_clean" in st.session_state else df_raw
        # Show preview 
        st.subheader("Preview of Your Data")
        if apply_cleaning and "df_clean" in st.session_state:
            st.dataframe(st.session_state["df_clean"].head(100))
        else:
            st.dataframe(df_raw.head(100))
        # Sample if needed
        if len(df_current) > 5000:
            st.warning(f"Large dataset detected ({len(df_current)} rows). Sampling 1000 rows for faster performance.")
            df_sample = df_current.sample(n=1000, random_state=42)
        else:
            df_sample = df_current
        # ✅ Save it for use in other tabs
        st.session_state["df_sample"] = df_sample
    
    
            
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
            
            drop_columns = st.multiselect("Optional: Drop Columns Before Processing", df_clean.columns.tolist())
    
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
                df_encoded = df_clean.drop(columns=drop_columns) if drop_columns else df_clean.copy()
        
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
with tab2:  
        with st.expander("💬 **Ask about your data**"):
            st.markdown("""
            - What are the key trends in this dataset?
            - What is the average sale price?
            - How many items were sold per region?
            - Which month had the highest revenue?
            - What insights can you provide from this data?
            """)
                
        # --- Chat Section ---
        user_question = st.text_input("Ask a question about your data:")
    
        # --- Get Cleaned Data ---
        df_clean = st.session_state.get("df_clean")
        df_sample = st.session_state.get("df_sample")
        
        if df_clean is not None:
            # --- Light Sampling for Large Files ---    
            if len(df_clean) > 5000:
                st.warning(f"Large dataset detected ({len(df_sample)} rows). Sampling 1000 rows for efficiency.")
                df_sample = df_sample.sample(n=1000, random_state=42)
            else:
                df_sample = df_sample
        
        
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
                                {"role": "system", "content": "You are a helpful Analyst. Be concise by default. Provide direct answers unless the user asks for explanation or calculation steps."},
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
            st.warning("Please upload and clean your dataset first in Tab 1.")
              
    
        # # --- Optional Chart Builder ---
        # with st.expander("🛠️ Create a Custom Chart (optional)"):
        #     chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter"])
        #     x_col = st.selectbox("Select X-axis column", options=df_sample.columns)
        #     y_col = st.selectbox("Select Y-axis column", options=df_sample.columns)
        #     if st.button("Generate Custom Chart"):
        #         try:
        #             if chart_type == "Bar":
        #                 fig = px.bar(df_sample, x=x_col, y=y_col)
        #             elif chart_type == "Line":
        #                 fig = px.line(df_sample, x=x_col, y=y_col)
        #             elif chart_type == "Scatter":
        #                 fig = px.scatter(df_sample, x=x_col, y=y_col)
        #             st.plotly_chart(fig)
        #         except Exception as e:
        #             st.error(f"Chart creation failed: {e}")
                
        # --- Divider --
        #st.markdown("---")            

    # --- Smart Auto Insights ---
with tab3:    
        with st.expander("Smart Auto Insights", expanded=True):
            st.markdown("Get an instant overview of your dataset without lifting a finger. This section auto-generates summaries, stats, and visuals for quick insight.")
            # --- Safety check ---
            if 'df_sample' in st.session_state and st.session_state['df_sample'] is not None:
                df_sample = st.session_state['df_sample']
                
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
        
                  #Divider
                st.markdown("---")
                
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
                
                    for i in range(0, len(numeric_cols), 2):  # Two charts per row
                        cols = st.columns(2)
                
                        for j in range(2):
                            if i + j < len(numeric_cols):
                                col = numeric_cols[i + j]
                                with cols[j]:
                                    try:
                                        #st.markdown(f"**{col}**")
                                        binned_col = pd.cut(df_sample[col], bins=10)
                                        counts = binned_col.value_counts().sort_index()
                                        # Clean bin labels by rounding left/right edges to whole numbers
                                        clean_bin_labels = [f"{int(interval.left)}–{int(interval.right)}" for interval in counts.index]
                                        vc_df = pd.DataFrame({f"{col} (binned)": clean_bin_labels, "Count": counts.values})
                
                                        if chart_type == "Bar (Counts)":
                                            fig = px.bar(vc_df, x=f"{col} (binned)", y="Count", title=f"{col}")
                                            fig.update_layout(
                                                title={
                                                    'text': f"{col} - Bar Chart (Binned)",
                                                    'x': 0.5,
                                                    'xanchor': 'center',
                                                    'font': dict(size=18)
                                                }
                                            )
                                        elif chart_type == "Line (Counts)":
                                            fig = px.line(vc_df, x=f"{col} (binned)", y="Count", title=f"{col}")
                                            fig.update_layout(
                                                title={
                                                    'text': f"{col} - Bar Chart (Binned)",
                                                    'x': 0.5,
                                                    'xanchor': 'center',
                                                    'font': dict(size=18)
                                                }
                                            )
                                        else:
                                            st.warning("Chart type not recognized.")
                
                                        st.plotly_chart(fig, use_container_width=True)
                                    except Exception as e:
                                        st.warning(f"Could not generate chart for {col}: {e}")
                    
                #Divider
                st.markdown("---")    
        
                st.subheader("Categorical Count Explorer")
                cat_cols = df_sample.select_dtypes(include='object').columns.tolist()
                
                if not cat_cols:
                    st.info("No categorical columns found in your dataset.")
                else:
                    selected_cat = st.selectbox("Select a categorical column", cat_cols)
                
                    if selected_cat:
                        top_n = st.slider("Limit to top N values", min_value=5, max_value=30, value=10)
                        count_df = df_sample[selected_cat].value_counts().head(top_n).reset_index()
                        count_df.columns = [selected_cat, "Count"]
                
                        chart_style = st.radio("Chart style:", ["Bar", "Pie"], horizontal=True)
                
                        st.dataframe(count_df)
                
                        if chart_style == "Bar":
                            fig = px.bar(count_df, x=selected_cat, y="Count", title=f"Top {top_n} {selected_cat}")
                        else:
                            fig = px.pie(count_df, names=selected_cat, values="Count", title=f"{selected_cat} Distribution")
                
                        st.plotly_chart(fig, use_container_width=True)
                        
        
                st.markdown("---")
                
                st.subheader("📸 Exportable Dashboard Snapshot (**BETA**)")
                if st.button("📥 Generate & Download Image Summary"):
                    try:
                        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
                        axes = axes.flatten()
                
                        # Sample preview
                        sns.heatmap(df_sample.head(10).isnull(), ax=axes[0], cbar=False)
                        axes[0].set_title("Missing Values (Top 10 Rows)")
                
                        # First numeric chart
                        if len(df_sample.select_dtypes(include="number").columns) > 0:
                            col1 = df_sample.select_dtypes(include="number").columns[0]
                            sns.histplot(df_sample[col1], ax=axes[1], kde=True)
                            axes[1].set_title(f"Distribution: {col1}")
                
                        # First object chart
                        if len(df_sample.select_dtypes(include="object").columns) > 0:
                            col2 = df_sample.select_dtypes(include="object").columns[0]
                            vc = df_sample[col2].value_counts().nlargest(5)
                            sns.barplot(x=vc.values, y=vc.index, ax=axes[2])
                            axes[2].set_title(f"Top Categories: {col2}")
                
                        # Forecast line (if exists)
                        if 'forecast_df' in locals():
                            axes[3].plot(forecast_df[date_col], forecast_df["Forecast"], marker='o')
                            axes[3].set_title("Forecast Preview")
                        else:
                            axes[3].axis('off')
                
                        plt.tight_layout()
                
                        # Save to BytesIO
                        buf = io.BytesIO()
                        plt.savefig(buf, format="png")
                        buf.seek(0)
                
                        b64 = base64.b64encode(buf.read()).decode()
                        href = f'<a href="data:file/png;base64,{b64}" download="dashboard_snapshot.png">📥 Click to download image</a>'
                        st.markdown(href, unsafe_allow_html=True)
                
                        st.success("✅ Snapshot ready!")
                    except Exception as e:
                        st.warning(f"Something went wrong: {e}")
        
    # --- Guidance for ML Tools --
with tab4:
    if "df_sample" not in st.session_state or st.session_state["df_sample"] is None:
        st.warning("⚠️ No dataset loaded yet. Please upload your file in Tab 1.")
    else:
        df_sample = st.session_state["df_sample"]
        
        #st.markdown("---")
        st.markdown("## 🔬 Forecast Modeling & Advanced Analysis")
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
                      
        # --- Advanced Forecasting with Prophet ---
        with st.expander("🔮 Advanced Forecasting (Prophet)", expanded=False):
            with st.expander("ℹ️ Prophet Forecasting Requirements", expanded=False):
                st.markdown("""
                - **Data must be time series** (e.g., monthly sales)
                - Minimum of **12 time points** for meaningful predictions
                - Prophet expects **consistent intervals** (no gaps)
                - Date column will be automatically converted to `ds`
                - Value to forecast will be used as `y`
                """)
            try:
                date_cols = [col for col in df_sample.columns if pd.api.types.is_datetime64_any_dtype(df_sample[col])]
                numeric_cols = df_sample.select_dtypes(include='number').columns.tolist()
        
                if not date_cols:
                    st.warning("No datetime column found. Please include a date column to enable Prophet forecasting.")
                else:
                    date_col = st.selectbox("📅 Select date column (Prophet):", date_cols, key="prophet_date")
                    target_col = st.selectbox("📈 Select value to forecast (Prophet):", numeric_cols, key="prophet_target")
                    forecast_months = st.slider("⏩ Months to forecast (Prophet)", 1, 12, 6, key="prophet_months")
        
                    df_prophet = df_sample[[date_col, target_col]].dropna().copy()
                    df_prophet.columns = ["ds", "y"]
                    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"], errors="coerce")
                    df_prophet = df_prophet.dropna()
        
                    m = Prophet()
                    m.fit(df_prophet)
        
                    future = m.make_future_dataframe(periods=forecast_months * 30, freq='D')  # roughly 1 month = 30 days
                    forecast = m.predict(future)
        
                    st.write("📊 Forecast Table:")
                    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_months * 30))
        
                    st.write("📈 Forecast Plot:")
                    fig1 = m.plot(forecast)
                    st.pyplot(fig1)
        
                    st.write("📉 Forecast Components:")
                    fig2 = m.plot_components(forecast)
                    st.pyplot(fig2)
        
            except Exception as e:
                st.error(f"Prophet forecasting failed: {e}")
        
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
with tab5:
        #st.markdown("---")
        st.markdown("## Data Science & Machine Learning Modeling")
        st.info("This section includes advanced machine learning tools for data scientists and experienced analysts.")
        # --- Data Status Bar ---
        df_clean = st.session_state.get("df_clean")
        df_sample = st.session_state.get("df_sample")
        df_norm = st.session_state.get("normalized_data")

        if df_sample is None:
            st.error("🚫 No dataset loaded. Please upload your data in Tab 1.")
            st.stop()
        else:
            st.success("✅ Dataset loaded.")
            if df_clean is not None:
                st.info("🧼 Cleaned data is being used.")
            else:
                st.warning("⚠️ Using raw (uncleaned) data.")
    
            if df_norm is not None:
                st.success("🧪 Normalized dataset will be used for ML modeling.")
            else:
                st.warning("⚠️ Normalized dataset not found. Please normalize your data in Tab 1 for better model performance.")
    
        # --- Advanced Data Scientist Tools (Expandable Section) ---
        with st.expander("🔬 Data Scientist Tools (Pro Preview) *Beta* ", expanded=False):
            data_for_modeling = st.session_state.get("normalized_data", df_sample)
        
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
                            _[Your explanation content]_
                            """)
        
                        if st.button("🌲 Run Random Forest Model"):
                            try:
                                # 🛠️ Hyperparameters
                                st.sidebar.header("🛠️ Model Settings")
                                n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 10, 500, 100, step=10)
                                max_depth = st.sidebar.slider("Max Depth", 1, 50, 10)
                                min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 5)
                                min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 20, 2)
        
                                # 📊 Data prep
                                X = data_for_modeling[features]
                                y = data_for_modeling[target_col]
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
                                # 🌲 Train model
                                model = RandomForestRegressor(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    max_features='sqrt',
                                    random_state=42,
                                    n_jobs=-1
                                )
                                model.fit(X_train, y_train)
        
                                # 🔍 Feature Importances
                                importances = model.feature_importances_
                                feature_df = pd.DataFrame({
                                    "Feature": features,
                                    "Importance": importances
                                }).sort_values(by="Importance", ascending=False)
                                st.subheader("🔍 Feature Importances")
                                st.dataframe(feature_df)
                                fig = px.bar(feature_df, x="Feature", y="Importance", title="Feature Importance (Random Forest)")
                                st.plotly_chart(fig)
        
                                # 🎯 Predictions
                                y_pred = model.predict(X_test)
                                sample_df = pd.DataFrame({
                                    "Actual": y_test.values,
                                    "Predicted": y_pred
                                }).reset_index(drop=True)
                                st.subheader("🎯 Prediction Samples (Actual vs. Predicted)")
                                st.dataframe(sample_df.head(10))
        
                                # 📈 Metrics
                                mae = mean_absolute_error(y_test, y_pred)
                                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                                r2 = r2_score(y_test, y_pred)
                                st.subheader("📈 Model Performance Metrics")
                                st.markdown(f"- **MAE:** `{mae:.2f}`")
                                st.markdown(f"- **RMSE:** `{rmse:.2f}`")
                                st.markdown(f"- **R² Score:** `{r2:.2f}`")
        
                                # 🔁 Cross-validation
                                with st.spinner("Running 5-fold Cross-Validation..."):
                                    cv_score = cross_val_score(model, X, y, cv=5, scoring='r2').mean()
                                st.markdown(f"- **Cross-Validated R² Score:** `{cv_score:.2f}`")
        
                                # ⚠️ Warnings
                                st.subheader("⚠️ Model Diagnostic")
                                if r2 < 0.2:
                                    st.warning("Model R² is low.")
                                elif r2 > 0.9 and (r2 - cv_score) > 0.1:
                                    st.warning("Possible overfitting.")
        
                                st.success("Random Forest model completed!")
        
                            except Exception as e:
                                st.error(f"❌ Error running model: {e}")
        
                except Exception as e:
                    st.error(f"❌ Error preparing data: {e}")
        
            else:
                st.info("Please upload a file to get started.")

            
# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ using Streamlit + OpenAI + Pandas + Plotly")
st.markdown("📬 Need help? Contact us at [pocketanalyst.help@gmail.com](mailto:pocketanalyst.help@gmail.com)")
