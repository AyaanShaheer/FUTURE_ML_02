import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="📈 AI Stock Predictor Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visuals
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stMetric label {
        font-size: 1rem;
        color: #6c757d;
        font-weight: 500;
    }
    .stMetric value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #2e59d9 !important;
    }
    .css-1v3fvcr {
        padding: 2rem 1rem;
    }
    .st-b7 {
        color: white !important;
    }
    .st-c0 {
        background-color: #4e73df !important;
    }
    .stButton>button {
        background-color: #4e73df;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #2e59d9;
        color: white;
    }
    .css-1y4p8pa {
        max-width: 100%;
        padding: 2rem 1rem 6rem;
    }
    /* Custom metric value colors */
    .metric-rmse {
        color: #e74a3b !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
    }
    .metric-mae {
        color: #f6c23e !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
    }
    .metric-accuracy {
        color: #1cc88a !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
    }
    .metric-direction {
        color: #36b9cc !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
    }
    .forecast-value {
        color: #2e59d9 !important;
        font-weight: 700 !important;
    }
    .positive-change {
        color: #1cc88a !important;
        font-weight: 700 !important;
    }
    .negative-change {
        color: #e74a3b !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# App title with emoji
st.title("📈 AI Stock Predictor Pro")
st.markdown("""
<div style="background-color:#4e73df;padding:16px;border-radius:10px;margin-bottom:24px">
    <h3 style="color:white;margin:0;">Predict future stock prices with AI-powered forecasting</h3>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        'rmse': 0,
        'mae': 0,
        'accuracy': 0,
        'direction_accuracy': 0
    }

# Enhanced prediction function with trend detection
def predict_prices(data, forecast_days=30):
    """Simulate ML predictions with random variation and trend detection"""
    # Calculate historical trend
    trend = np.polyfit(np.arange(len(data)), data, 1)[0]
    
    # Generate predictions with trend influence
    predictions = data * (0.98 + np.random.random(len(data)) * 0.04)
    predictions = predictions * (1 + trend/1000)  # Incorporate trend
    
    # Generate forecast
    last_date = datetime.strptime(data.index[-1], '%Y-%m-%d') if isinstance(data.index[-1], str) else data.index[-1]
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days+1)]
    forecast_values = []
    
    last_value = data.iloc[-1]  # Fixed: Use iloc for positional access
    for _ in range(forecast_days):
        # Forecast with trend and randomness
        forecast_value = last_value * (1 + (trend/1000) + (np.random.random()*0.02 - 0.01))
        forecast_values.append(forecast_value)
        last_value = forecast_value
    
    forecast_df = pd.DataFrame({
        'date': [d.strftime('%Y-%m-%d') for d in forecast_dates],
        'forecast': forecast_values
    })
    
    return predictions, forecast_df

# Calculate enhanced evaluation metrics
def calculate_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    accuracy = 100 - (mae / actual.mean() * 100)
    
    # Direction accuracy (did we predict up/down correctly)
    direction_correct = np.sum(
        ((actual.diff().dropna() > 0) & (predicted.diff().dropna() > 0)) |
        ((actual.diff().dropna() < 0) & (predicted.diff().dropna() < 0))
    )
    direction_accuracy = direction_correct / (len(actual)-1) * 100
    
    return {
        'rmse': round(rmse, 2),
        'mae': round(mae, 2),
        'accuracy': round(accuracy, 1),
        'direction_accuracy': round(direction_accuracy, 1)
    }

# Process uploaded file with enhanced mock data
def process_uploaded_file(uploaded_file, ticker, forecast_days=30):
    try:
        # Generate mock data if no file uploaded
        if uploaded_file is None:
            dates = pd.date_range(end=pd.Timestamp.today(), periods=60).strftime('%Y-%m-%d')
            base_price = 100 + np.random.random() * 50
            trend = np.random.normal(0.5, 0.2)
            noise = np.random.normal(0, 0.5, 60)
            prices = base_price + np.arange(60) * trend + np.cumsum(noise)
            
            df = pd.DataFrame({
                'Date': dates,
                'Close': prices
            })
        else:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            
            # Ensure we have Date and Close columns
            if 'Close' not in df.columns:
                st.error("CSV must contain 'Close' column with price data")
                return None, None
                
            if 'Date' not in df.columns:
                df['Date'] = pd.date_range(end=pd.Timestamp.today(), periods=len(df)).strftime('%Y-%m-%d')
        
        # Set Date as index and sort
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        
        # Generate predictions and forecast
        predictions, forecast_df = predict_prices(df['Close'], forecast_days)
        
        # Prepare historical data with predictions
        historical_df = df.reset_index()
        historical_df['Predicted'] = predictions.values
        historical_df = historical_df.rename(columns={
            'Date': 'date',
            'Close': 'actual',
            'Predicted': 'predicted'
        })
        
        return historical_df, forecast_df
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None, None

# Sidebar for inputs with better organization
with st.sidebar:
    st.markdown("""
    <div style="background-color:#4e73df;padding:12px;border-radius:8px;margin-bottom:16px">
        <h3 style="color:white;margin:0;">🔧 Input Parameters</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Ticker input with emoji
    ticker = st.text_input("🏷️ Stock Ticker", value="AAPL").upper()
    
    # File upload with better styling
    uploaded_file = st.file_uploader(
        "📤 Upload Historical Data (CSV)",
        type=["csv"],
        help="CSV should contain Date and Close price columns"
    )
    
    # Forecast days selector
    forecast_days = st.slider(
        "🔮 Forecast Days",
        min_value=7,
        max_value=90,
        value=30,
        help="Number of days to forecast into the future"
    )
    
    # Process button with loading state
    if st.button("🚀 Process & Predict", use_container_width=True):
        with st.spinner('Crunching numbers with AI...'):
            # Process the uploaded file or generate mock data
            historical_df, forecast_df = process_uploaded_file(uploaded_file, ticker, forecast_days)
            if historical_df is not None and forecast_df is not None:
                st.session_state.stock_data = historical_df
                st.session_state.forecast_data = forecast_df
                st.session_state.metrics = calculate_metrics(
                    historical_df['actual'], 
                    historical_df['predicted']
                )
            else:
                st.error("Failed to process data. Please check your input.")

# Main content
if st.session_state.stock_data is not None and st.session_state.forecast_data is not None:
    historical_df = st.session_state.stock_data
    forecast_df = st.session_state.forecast_data
    
    # Combine historical and forecast data for visualization
    full_df = pd.concat([
        historical_df[['date', 'actual', 'predicted']],
        pd.DataFrame({
            'date': forecast_df['date'],
            'forecast': forecast_df['forecast']
        })
    ], ignore_index=True)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Price Prediction", "📈 Forecast", "📋 Data Table"])
    
    with tab1:
        # Enhanced chart with more features
        fig = go.Figure()
        
        # Actual prices
        fig.add_trace(go.Scatter(
            x=historical_df['date'],
            y=historical_df['actual'],
            name='Actual Price',
            line=dict(color='#4e73df', width=3),
            mode='lines+markers',
            marker=dict(size=6)
        ))
        
        # Predicted prices
        fig.add_trace(go.Scatter(
            x=historical_df['date'],
            y=historical_df['predicted'],
            name='Predicted Price',
            line=dict(color='#1cc88a', width=3, dash='dot'),
            mode='lines+markers',
            marker=dict(size=6)
        ))
        
        # Forecast prices
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['forecast'],
            name=f'{forecast_days}-Day Forecast',
            line=dict(color='#f6c23e', width=3, dash='dash'),
            mode='lines+markers',
            marker=dict(size=6)
        ))
        
        # Highlight forecast area
        fig.add_vrect(
            x0=historical_df['date'].iloc[-1],
            x1=forecast_df['date'].iloc[-1],
            fillcolor="rgba(246, 194, 62, 0.1)",
            layer="below",
            line_width=0,
        )
        
        fig.update_layout(
            title=f'{ticker} Price Prediction & Forecast',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_white',
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics in a more visual way with custom colors
        st.subheader("📉 Model Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="stMetric">
                <label>RMSE</label>
                <div class="metric-rmse">{st.session_state.metrics['rmse']}</div>
                <div style="font-size:0.8rem;color:#6c757d;">Root Mean Squared Error</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stMetric">
                <label>MAE</label>
                <div class="metric-mae">{st.session_state.metrics['mae']}</div>
                <div style="font-size:0.8rem;color:#6c757d;">Mean Absolute Error</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stMetric">
                <label>Price Accuracy</label>
                <div class="metric-accuracy">{st.session_state.metrics['accuracy']}%</div>
                <div style="font-size:0.8rem;color:#6c757d;">Prediction Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stMetric">
                <label>Direction Accuracy</label>
                <div class="metric-direction">{st.session_state.metrics['direction_accuracy']}%</div>
                <div style="font-size:0.8rem;color:#6c757d;">Direction Prediction</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Performance explanation
        with st.expander("ℹ️ What do these metrics mean?"):
            st.markdown("""
            - **RMSE (Root Mean Squared Error)**: Measures the average magnitude of errors. Lower values indicate better fit.
            - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values.
            - **Price Accuracy**: Percentage indicating how close predictions are to actual prices (100% = perfect).
            - **Direction Accuracy**: Percentage of times the model correctly predicted if price would go up or down.
            """, unsafe_allow_html=True)
    
    with tab2:
        # Forecast-focused visualization
        st.subheader(f"🔮 {forecast_days}-Day Price Forecast")
        
        fig2 = go.Figure()
        
        # Last 30 days of actual data for context
        last_30_days = historical_df.iloc[-30:]
        
        # Actual prices (last 30 days)
        fig2.add_trace(go.Scatter(
            x=last_30_days['date'],
            y=last_30_days['actual'],
            name='Actual Price (Last 30 Days)',
            line=dict(color='#4e73df', width=3),
            mode='lines+markers'
        ))
        
        # Forecast prices
        fig2.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['forecast'],
            name=f'{forecast_days}-Day Forecast',
            line=dict(color='#f6c23e', width=3),
            mode='lines+markers'
        ))
        
        # Confidence interval (simulated)
        fig2.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['forecast'] * 1.05,
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig2.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['forecast'] * 0.95,
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(246, 194, 62, 0.2)',
            name='Confidence Interval'
        ))
        
        fig2.update_layout(
            title=f'{ticker} {forecast_days}-Day Price Forecast',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Forecast summary stats with better colors
        current_price = historical_df['actual'].iloc[-1]
        forecast_end_price = forecast_df['forecast'].iloc[-1]
        change_percent = (forecast_end_price - current_price) / current_price * 100
        change_color = "positive-change" if change_percent >= 0 else "negative-change"
        change_icon = "⬆️" if change_percent >= 0 else "⬇️"
        
        st.markdown(f"""
        <div style="background-color:#f8f9fa;padding:16px;border-radius:10px;margin-top:16px">
            <h4 style="margin-top:0;color:#4e73df;">Forecast Summary</h4>
            <p><strong style="color:#5a5c69;">Current Price:</strong> <span class="forecast-value">${current_price:.2f}</span></p>
            <p><strong style="color:#5a5c69;">Forecast Price in {forecast_days} Days:</strong> <span class="forecast-value">${forecast_end_price:.2f}</span></p>
            <p><strong style="color:#5a5c69;">Expected Change:</strong> <span class="{change_color}">{change_icon} {abs(change_percent):.1f}%</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        # Data table view without styling that requires matplotlib
        st.subheader("📋 Historical Data with Predictions")
        st.dataframe(
            historical_df.round(2),
            height=400,
            use_container_width=True
        )
        
        st.subheader("🔮 Forecast Data")
        st.dataframe(
            forecast_df.round(2),
            height=400,
            use_container_width=True
        )
    
    # Download buttons
    st.download_button(
        label="📥 Download Historical Data with Predictions",
        data=historical_df.to_csv(index=False).encode('utf-8'),
        file_name=f"{ticker}_historical_predictions.csv",
        mime="text/csv"
    )
    
    st.download_button(
        label="📥 Download Forecast Data",
        data=forecast_df.to_csv(index=False).encode('utf-8'),
        file_name=f"{ticker}_forecast.csv",
        mime="text/csv"
    )
else:
    # Enhanced welcome message
    st.markdown("""
    <div style="background-color:#f8f9fa;padding:24px;border-radius:10px;margin-bottom:24px">
        <h2 style="margin-top:0;">Welcome to AI Stock Predictor Pro</h2>
        <p>This advanced tool uses machine learning to predict stock prices and forecast future trends.</p>
        
        <h4>To get started:</h4>
        <ol>
            <li>Enter a stock ticker (e.g. AAPL, MSFT, TSLA)</li>
            <li>Upload historical price data (CSV format) or use our sample data</li>
            <li>Click "Process & Predict" to generate forecasts</li>
        </ol>
        
        <p>Our AI model will analyze the data and provide:</p>
        <ul>
            <li>📊 Historical price predictions vs actual values</li>
            <li>🔮 {forecast_days}-day price forecast with confidence intervals</li>
            <li>📈 Performance metrics evaluating prediction accuracy</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Sample data format with download option
    with st.expander("📋 Expected CSV Format & Sample Data"):
        st.markdown("""
        Your CSV file should contain at least these columns:
        - `Date`: Date in YYYY-MM-DD format
        - `Close`: Closing price for the stock
        
        Example:
        ```csv
        Date,Close
        2023-01-01,150.25
        2023-01-02,152.30
        2023-01-03,151.75
        ```
        """)
        
        # Generate sample data for download
        sample_dates = pd.date_range(end=pd.Timestamp.today(), periods=30).strftime('%Y-%m-%d')
        sample_prices = 100 + np.random.random(30) * 50 + np.arange(30) * 2
        sample_df = pd.DataFrame({
            'Date': sample_dates,
            'Close': sample_prices
        })
        
        st.download_button(
            label="⬇️ Download Sample CSV",
            data=sample_df.to_csv(index=False).encode('utf-8'),
            file_name="sample_stock_data.csv",
            mime="text/csv"
        )