# 📈 Sales Forecasting Dashboard

An interactive Streamlit web application to **forecast monthly sales** using two powerful machine learning models:  
- **Prophet** (for trend & seasonality)
- **XGBoost** (for feature-based nonlinear modeling)

This dashboard helps business users explore trends, compare forecasts, and make data-driven decisions.

---

## ✨ **Features**
- Interactive **Plotly charts** and tables
- Forecast **future months** with a slider
- Clean, professional UI with sidebar controls
- Compare Prophet vs. XGBoost predictions
- Historical sales visualization

---

## 🚀 **How it works**
✅ Load and preprocess real sales data  
✅ Use saved trained models:
- `prophet_model.pkl`
- `xgboost_sales_forecast.pkl`
✅ Generate forecasts for next N months
✅ Display results in dynamic tables & plots

---

## 🛠 **How to run locally**
```bash
# Clone the repository
git clone https://github.com/yourusername/sales-forecast-app.git
cd sales-forecast-app

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
