

---

# 📈 AI Stock Predictor Pro

*A Streamlit-based AI-powered app for predicting stock prices and forecasting future trends.*

![Banner](https://github.com/user-attachments/assets/2b3887a7-38d6-4495-b5ec-720a9ea41a5c)

---

## 🚀 Overview

**AI Stock Predictor Pro** is a lightweight, interactive tool for stock price prediction and forecasting using historical stock data. Built with **Streamlit**, it offers real-time visualizations, forecasting with confidence intervals, and downloadable reports — all in your browser.

> 💡 Ideal for data science learners, finance enthusiasts, and developers exploring time series forecasting!

---

## 🧠 Features

* 📂 **Upload Historical Data**: Load your own stock CSV or use the built-in sample.
* 📉 **AI-Based Predictions**: Compare actual vs predicted stock prices.
* ⏩ **Future Forecasting**: Generate 30-day forecasts with confidence intervals.
* 📊 **Performance Metrics**: Get RMSE, MAE, Price Accuracy, and Direction Accuracy.
* 📈 **Interactive Visuals**: Explore trends with Plotly-powered charts.
* 📥 **Download Results**: Export prediction and forecast data as CSV.

---

## 🖼️ App Previews

| 📊 Model Performance Metrics                                                                              | 🔮 Forecasting                                                                                   | 📋 Analaysis                                                                                  |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| ![Screenshot 1](https://github.com/user-attachments/assets/ca17a174-219b-4031-acd7-ecade701dc5f) | ![Screenshot 2](https://github.com/user-attachments/assets/82c1ee04-ac36-4556-947d-cede66722b03) | ![Screenshot 3](https://github.com/user-attachments/assets/2b3887a7-38d6-4495-b5ec-720a9ea41a5c) |

---

## 🛠️ Installation

### ⚙️ Prerequisites

* Python 3.7+
* Git

### 📦 Setup

```bash
# Clone the repository
git clone https://github.com/AyaanShaheer/FUTURE_ML_02.git
cd stock_predictor

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🌐 Usage Guide

1. Open the app in your browser (Streamlit will give a link like `http://localhost:8501`).
2. Enter a stock ticker (e.g., `AAPL`) or upload your own `.csv` file.
3. Adjust the **forecast window** (default is 30 days).
4. Click **Process & Predict**.
5. Navigate through tabs:

   * **Price Prediction**
   * **Forecast**
   * **Data Table**
6. Export results using **Download CSV** buttons.

---

## 📄 CSV Format

Your file must include:

| Date (YYYY-MM-DD) | Close (Price) |
| ----------------- | ------------- |
| 2023-01-01        | 150.25        |
| 2023-01-02        | 152.30        |
| ...               | ...           |

> ✅ Use the included `sample_stock_data.csv` for testing.

---

## 🧱 Project Structure

```
stock_predictor/
├── app.py               # Streamlit app
├── requirements.txt     # Dependencies
├── sample_stock_data.csv
└── README.md
```

---

## 🧪 Tech Stack

* `Streamlit` — UI & dashboard
* `pandas`, `numpy` — Data processing
* `scikit-learn` — Metrics (RMSE, MAE, etc.)
* `plotly` — Interactive visualizations

---

## ⚠️ Notes

* The prediction logic currently simulates trends with random variation for demo purposes.
* For production, replace the logic in `predict_prices()` with models like **Prophet**, **ARIMA**, **LSTM**, or **XGBoost**.
* This app is for **educational purposes only** and should not be used for real financial decisions.

---

## 📬 Contact

Have questions or suggestions?
📧 Email: [gfever@example.com](mailto:gfever252@gmail.com)
💬 Or open an issue [here](https://github.com/AyaanShaheer/FUTURE_ML_02/issues)

---

## 🤝 Contributing

Contributions are welcome!

```bash
# Fork the repository
# Make your changes
# Submit a pull request 🚀
```

---

## 📄 License

Licensed under the [MIT License](LICENSE).

---


