import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(page_title="Sales Forecasting", layout="wide")

# ------------------ UI STYLE ------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #eaf4ff;
}
h1, h2, h3, h4, p, label {
    color: black !important;
}
[data-testid="stFileUploader"] {
    background-color: #2b2d36;
    padding: 12px;
    border-radius: 12px;
}
[data-testid="stFileUploader"] button {
    background-color: #e5e7eb !important;
    color: black !important;
    border-radius: 10px !important;
    padding: 6px 16px;
}
[data-testid="stFileUploader"] label {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.title("📊 Sales Forecasting System")

# ------------------ UPLOAD ------------------
st.markdown("<h3 style='color:white;'>📁 Upload Folder</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(" ", type=['csv'])

if uploaded_file:

    # ------------------ LOAD DATA ------------------
    df = pd.read_csv(uploaded_file, encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df = df.sort_values(by='Order Date')

    # ------------------ GROUP DATA ------------------
    df_grouped = df.groupby('Order Date')['Sales'].sum()
    df_grouped = df_grouped.asfreq('D').fillna(0)

    # ------------------ KPI ------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Total Sales", int(df_grouped.sum()))
    col2.metric("📈 Avg Sales", int(df_grouped.mean()))
    col3.metric("📅 Total Days", len(df_grouped))

    # ------------------ MAIN GRAPH ------------------
    fig = px.line(
        df_grouped,
        x=df_grouped.index,
        y='Sales',
        title="📈 Sales Trend",
        markers=True
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified"
    )

    # Stock-style filters (fixed)
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=7, label="1W", step="day", stepmode="backward"),
                dict(count=30, label="1M", step="day", stepmode="backward"),
                dict(count=90, label="3M", step="day", stepmode="backward"),
                dict(count=180, label="6M", step="day", stepmode="backward"),
                dict(step="all")
            ]
        ),
        rangeslider=dict(visible=True)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------ SEASONAL DECOMPOSITION ------------------
    st.subheader("📊 Time Series Decomposition")

    decomposition = seasonal_decompose(df_grouped, model='additive', period=30)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    st.markdown("### 📈 Trend")
    st.line_chart(trend.dropna(), use_container_width=True)

    st.markdown("### 🔁 Seasonal Pattern")
    st.line_chart(seasonal.dropna(), use_container_width=True)

    st.markdown("### ⚡ Residual (Noise)")
    st.line_chart(residual.dropna(), use_container_width=True)

    # ------------------ FORECAST ------------------
    st.subheader("🔮 Future Sales Prediction")

    model = ARIMA(df_grouped, order=(5,1,0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=10)

    forecast_df = forecast.reset_index()
    forecast_df.columns = ["Date", "Predicted Sales"]

    st.dataframe(forecast_df)