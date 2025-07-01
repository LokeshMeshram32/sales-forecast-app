# 📈 Sales Forecasting Dashboard

An interactive Streamlit web application to **forecast monthly sales** using two machine learning models:
- **Prophet** (captures trend & seasonality)
- **XGBoost** (models nonlinear patterns with engineered features)

This dashboard helps business users explore trends, compare forecasts, and make data-driven decisions in real time.

---

## ✨ **Features**
✅ Dynamic Plotly charts & tables  
✅ Forecast next N months with a slider  
✅ Compare Prophet vs. XGBoost predictions side-by-side  
✅ Clear, business-friendly UI with sidebar controls  
✅ Historical sales visualization

---

## 🌐 **Live App**
[▶️ Click here to explore the app!](https://sales-forecast-app-e2uqbsvbevauwwagzzzecs.streamlit.app/)

*(Hosted on Streamlit Cloud)*

---

## 🚀 **How it works**
- Loads & preprocesses real sales data (`sales_data_sample.csv`)
- Uses saved models:
  - `prophet_model.pkl`
  - `xgboost_sales_forecast.pkl`
- Forecasts future monthly sales based on selected horizon
- Displays results in interactive Plotly tables & charts

---

## 🛠 **Run locally**
```bash
# Clone this repository
git clone https://github.com/yourusername/sales-forecast-app.git
cd sales-forecast-app

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
