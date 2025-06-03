AI Stock Predictor Pro
A Streamlit-based application for predicting stock prices and forecasting future trends using historical data. This tool provides interactive visualizations, performance metrics, and downloadable data for users to analyze stock price predictions.
Features

Upload Historical Data: Upload your stock data in CSV format or use the provided sample data.
Price Predictions: Compare actual stock prices with AI-predicted prices.
Future Forecasting: Generate 30-day price forecasts with confidence intervals.
Performance Metrics: Evaluate the model with metrics like RMSE, MAE, Price Accuracy, and Direction Accuracy.
Interactive Charts: Visualize historical predictions and forecasts using Plotly.
Downloadable Data: Export historical predictions and forecast data as CSV files.

Some Snippets from the App:

![Screenshot 2025-06-03 184143](https://github.com/user-attachments/assets/2b3887a7-38d6-4495-b5ec-720a9ea41a5c)

![Screenshot 2025-06-03 184151](https://github.com/user-attachments/assets/ca17a174-219b-4031-acd7-ecade701dc5f)

![Screenshot 2025-06-03 184203](https://github.com/user-attachments/assets/82c1ee04-ac36-4556-947d-cede66722b03)


Python 3.7 or higher
Git (to clone the repository)

Installation

Clone the repository:git clone https://github.com/AyaanShaheer/FUTURE_ML_02.git
cd stock_predictor


Create and activate a virtual environment:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install the required dependencies:pip install -r requirements.txt


Run the Streamlit app:streamlit run app.py



Usage

Open the app in your browser (Streamlit will provide the URL, typically http://localhost:8501).
Enter a stock ticker (e.g., AAPL).
Optionally, upload a CSV file with historical stock data (must include Date and Close columns). If no file is uploaded, the app uses sample data.
Adjust the forecast period using the slider (default: 30 days).
Click Process & Predict to generate predictions and forecasts.
Explore the results across three tabs:
Price Prediction: View historical predictions vs. actual prices.
Forecast: See the 30-day forecast with confidence intervals.
Data Table: Inspect historical and forecast data in tabular form.


Download the results as CSV files using the provided buttons.

CSV Format
Your CSV file should contain at least these columns:

Date: Date in YYYY-MM-DD format
Close: Closing price of the stock

Example:
Date,Close
2023-01-01,150.25
2023-01-02,152.30
2023-01-03,151.75

A sample CSV file (sample_stock_data.csv) is included in the repository for testing.
Project Structure
stock_predictor/
├── app.py                    # Main Streamlit application
├── requirements.txt          # List of dependencies
├── README.md                 # Project documentation


Dependencies
The project uses the following Python libraries (listed in requirements.txt):

streamlit: For the web application interface
pandas: For data manipulation
numpy: For numerical computations
plotly: For interactive visualizations
scikit-learn: For calculating performance metrics

Notes

The app currently uses a simulated prediction model with random variations and trend detection. For production use, consider replacing the predict_prices function with a real machine learning model (e.g., using tensorflow, pytorch, or prophet).
Ensure your CSV file matches the expected format to avoid errors.
The app is designed for educational purposes and should not be used for actual investment decisions.

License
This project is licensed under the MIT License. See the LICENSE file for details (if you add one to your repository).
Contributing
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.
Contact
For questions or feedback, feel free to open an issue on GitHub or contact me at your-email@example.com (replace with your email if desired).
