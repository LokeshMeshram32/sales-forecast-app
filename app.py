import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(page_title='📈 Sales Forecast Dashboard', page_icon='📊', layout='wide')

# Sidebar
st.sidebar.title('🔧 Forecast Settings')
n_months = st.sidebar.slider('Forecast next N months', min_value=1, max_value=12, value=3)
show_actuals = st.sidebar.checkbox('Show historical sales chart', value=True)
st.sidebar.markdown('---')
st.sidebar.caption('📊 *Built by Lokesh Meshram*')

# Title & intro
st.title('📈 Sales Forecasting Dashboard')
st.markdown("""
Interactive dashboard to forecast future monthly sales using **Prophet** and **XGBoost** models.  
Explore trends, compare models, and gain actionable business insights.
""")

# Load models & data
prophet_model = joblib.load('prophet_model.pkl')
xgb_model = joblib.load('xgboost_sales_forecast.pkl')

df = pd.read_csv('sales_data_sample.csv', encoding='latin1')
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
df['Month'] = df['ORDERDATE'].dt.to_period('M')
monthly_sales = df.groupby('Month')['SALES'].sum().reset_index()
monthly_sales['Month'] = monthly_sales['Month'].dt.to_timestamp()
monthly_sales['Month_num'] = np.arange(len(monthly_sales)) + 1
monthly_sales['Month_of_year'] = monthly_sales['Month'].dt.month
monthly_sales['Quarter'] = monthly_sales['Month'].dt.quarter

# Historical sales chart
if show_actuals:
    st.subheader('📅 Historical Monthly Sales')
    fig_hist = px.line(monthly_sales, x='Month', y='SALES',
                       title='Total Sales Over Time',
                       markers=True, template='plotly_white')
    fig_hist.update_traces(line=dict(color='royalblue'))
    st.plotly_chart(fig_hist, use_container_width=True)

# Forecast section
st.markdown('---')
st.subheader('🔮 Generate Forecast')

if st.button('Run Forecast'):
    # Prophet forecast
    future_dates = prophet_model.make_future_dataframe(periods=n_months, freq='M')
    forecast = prophet_model.predict(future_dates)
    prophet_forecast = forecast[['ds', 'yhat']].tail(n_months)
    prophet_forecast.rename(columns={'ds': 'Month', 'yhat': 'Predicted_Sales'}, inplace=True)

    # XGBoost forecast
    last_month_num = monthly_sales['Month_num'].max()
    last_month = monthly_sales['Month'].max()
    xgb_preds, xgb_dates = [], []

    for i in range(1, n_months+1):
        month_num = last_month_num + i
        month = (last_month + pd.DateOffset(months=i)).to_period('M').to_timestamp()
        month_of_year = month.month
        quarter = (month.month - 1)//3 + 1
        features = [month_num] + [1 if m==month_of_year else 0 for m in range(1,13)] + [1 if q==quarter else 0 for q in range(1,5)]
        pred = xgb_model.predict([features])[0]
        xgb_preds.append(pred)
        xgb_dates.append(month)

    xgb_df = pd.DataFrame({'Month': xgb_dates, 'Predicted_Sales': xgb_preds})

    # Fix: Convert all to lists of strings for Plotly tables
    prophet_months = prophet_forecast['Month'].dt.strftime('%b-%Y').tolist()
    prophet_sales = [f"${v:,.0f}" for v in prophet_forecast['Predicted_Sales'].tolist()]

    xgb_months = pd.Series(xgb_dates).dt.strftime('%b-%Y').tolist()
    xgb_sales = [f"${v:,.0f}" for v in xgb_preds]

    # Tabs for clean layout
    tab1, tab2, tab3 = st.tabs(["📦 Prophet Forecast", "📦 XGBoost Forecast", "📊 Comparison Chart"])

    with tab1:
        st.markdown('**Prophet Forecast Table**')
        fig_table_p = go.Figure(data=[go.Table(
            header=dict(values=['Month', 'Predicted Sales'],
                        fill_color='lightblue', font=dict(color='black', size=13)),
            cells=dict(values=[prophet_months, prophet_sales],
                       fill_color='white', font=dict(color='black'))
        )])
        fig_table_p.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_table_p, use_container_width=True)

    with tab2:
        st.markdown('**XGBoost Forecast Table**')
        fig_table_x = go.Figure(data=[go.Table(
            header=dict(values=['Month', 'Predicted Sales'],
                        fill_color='lightgreen', font=dict(color='black', size=13)),
            cells=dict(values=[xgb_months, xgb_sales],
                       fill_color='white', font=dict(color='black'))
        )])
        fig_table_x.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_table_x, use_container_width=True)

    with tab3:
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(x=prophet_forecast['Month'], y=prophet_forecast['Predicted_Sales'],
                                         mode='lines+markers', name='Prophet', line=dict(color='orange')))
        fig_compare.add_trace(go.Scatter(x=xgb_df['Month'], y=xgb_df['Predicted_Sales'],
                                         mode='lines+markers', name='XGBoost', line=dict(color='green')))
        fig_compare.update_layout(title='Forecast Comparison',
                                  xaxis_title='Month', yaxis_title='Predicted Sales',
                                  template='plotly_white', legend=dict(orientation='h', x=0.3, y=1.1))
        st.plotly_chart(fig_compare, use_container_width=True)

    st.success('✅ Forecast generated successfully!')

# Footer
st.markdown("""---  
✅ *Built with Streamlit, Prophet, XGBoost & Plotly*  
""")
