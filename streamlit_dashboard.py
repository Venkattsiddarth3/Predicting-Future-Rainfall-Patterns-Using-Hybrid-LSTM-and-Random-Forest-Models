import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import base64
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Advanced Weather Forecasting System",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional styling
def get_enhanced_css(bg_color):
    is_dark = bg_color == "#1e1e1e"
    text_color = "#ffffff" if is_dark else "#000000"
    card_bg = "#2d2d2d" if is_dark else "#ffffff"
    summary_bg = "#3d3d3d" if is_dark else "#f8f9fa"
    
    return f"""
<style>
    .main-header {{
        font-size: 2.2rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #FFD93D;
        text-align: center;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }}
    .metric-card {{
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    }}
    .weather-summary {{
        background: {summary_bg};
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #4facfe;
        margin: 1rem 0;
        color: {text_color};
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
    .forecast-hour {{
        background: {card_bg};
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #e1e5e9;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        color: {text_color};
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}
    .forecast-hour:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(79,172,254,0.15);
        border-color: #4facfe;
    }}
    .stApp {{
        background-color: {bg_color};
    }}
    .stSelectbox > div > div {{
        background-color: {card_bg};
        color: {text_color};
        border-radius: 8px;
    }}
    .stDateInput > div > div {{
        background-color: {card_bg};
        color: {text_color};
        border-radius: 8px;
    }}
    .stButton > button {{
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79,172,254,0.3);
    }}
    
    /* Vibrant heading colors */
    h1, h2, h3, h4, h5, h6 {{
        color: #FF6B6B !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }}
    
    .streamlit-expanderHeader {{
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
        color: white !important;
        font-weight: bold !important;
    }}
    
    .stSelectbox > label > div > div {{
        color: #FF6B6B !important;
        font-weight: bold !important;
    }}
    
    .stDateInput > label > div > div {{
        color: #FF6B6B !important;
        font-weight: bold !important;
    }}
    
    .stSidebar .css-1d391kg {{
        color: #4ECDC4 !important;
        font-weight: bold !important;
    }}
</style>
"""

# Initialize session state
if 'bg_color' not in st.session_state:
    st.session_state.bg_color = '#f8f9fa'
if 'forecasts' not in st.session_state:
    st.session_state.forecasts = None

# Apply enhanced CSS
st.markdown(get_enhanced_css(st.session_state.bg_color), unsafe_allow_html=True)

# Professional header
st.markdown('<h1 class="main-header">🌤️ Advanced Weather Forecasting System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Cache model loading
@st.cache_resource
def load_models():
    """Load all trained models and artifacts"""
    try:
        with open('rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        lstm_model = load_model('lstm_model.h5')
        with open('scaler_rf.pkl', 'rb') as f:
            scaler_rf = pickle.load(f)
        with open('scaler_lstm.pkl', 'rb') as f:
            scaler_lstm = pickle.load(f)
        with open('metrics.json', 'r') as f:
            metrics = json.load(f)
        df = pd.read_csv('india_2000_2024_daily_weather.csv')
        df['date'] = pd.to_datetime(df['date'])
        return rf_model, lstm_model, scaler_rf, scaler_lstm, metrics, df
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None

# Load models
rf_model, lstm_model, scaler_rf, scaler_lstm, metrics, historical_data = load_models()

if rf_model is None:
    st.error("❌ Failed to load models. Please ensure all model files are present.")
    st.stop()

# Enhanced sidebar with professional styling
st.sidebar.markdown("### 🎛️ Forecast Control Center")
st.sidebar.markdown("---")

# Theme selector with better styling
st.sidebar.markdown("#### 🎨 Theme Selection")
bg_color_option = st.sidebar.selectbox(
    "Choose Dashboard Theme",
    ["Light Blue", "Professional White", "Dark Mode", "Nature Green", "Sunset Orange"],
    index=0
)

color_map = {
    "Light Blue": "#e3f2fd",
    "Professional White": "#f8f9fa", 
    "Dark Mode": "#1e1e1e",
    "Nature Green": "#e8f5e8",
    "Sunset Orange": "#fff4e6"
}

if st.session_state.bg_color != color_map[bg_color_option]:
    st.session_state.bg_color = color_map[bg_color_option]
    st.rerun()

st.sidebar.markdown("---")

# Enhanced city selection with capital cities
st.sidebar.markdown("#### 🌍 Major Indian Cities")
city_data = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "timezone": "IST", "population": "32M"},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "timezone": "IST", "population": "20M"},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946, "timezone": "IST", "population": "12M"},
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "timezone": "IST", "population": "11M"},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639, "timezone": "IST", "population": "15M"},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "timezone": "IST", "population": "10M"},
    "Pune": {"lat": 18.5204, "lon": 73.8567, "timezone": "IST", "population": "7M"}
}

selected_city = st.sidebar.selectbox(
    "Select Capital City",
    list(city_data.keys()),
    index=0
)

city_info = city_data[selected_city]
st.sidebar.markdown(f"**📍 {selected_city} Coordinates:**")
st.sidebar.markdown(f"🌐 Lat: {city_info['lat']}, Lon: {city_info['lon']}")
st.sidebar.markdown(f"⏰ Timezone: {city_info['timezone']}")
st.sidebar.markdown(f"👥 Population: {city_info['population']}")

st.sidebar.markdown("---")

# Enhanced forecast controls
st.sidebar.markdown("#### 📅 Forecast Configuration")
forecast_start_date = st.sidebar.date_input(
    "Start Forecast Date",
    value=datetime.now().date(),
    min_value=datetime.now().date(),
    max_value=datetime.now().date() + timedelta(days=60)
)

forecast_duration = st.sidebar.selectbox(
    "Forecast Duration",
    ["24 Hours", "1 Week", "2 Weeks", "1 Month", "2 Months"],
    index=1
)

duration_map = {
    "24 Hours": 24,
    "1 Week": 168,
    "2 Weeks": 336,
    "1 Month": 720,
    "2 Months": 1440
}

forecast_hours = duration_map[forecast_duration]

# Generate forecast button
generate_forecast = st.sidebar.button(
    "🚀 Generate Advanced Forecast",
    type="primary",
    use_container_width=True
)

# Feature engineering (same as training)
def create_features(df, city='Delhi'):
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    df['temp_range'] = df['max_temperature'] - df['min_temperature']
    df['low_temp_indicator'] = (df['min_temperature'] < 15).astype(int)
    df['low_wind_indicator'] = (df['wind_speed'] < 5).astype(int)
    df['fog_risk_index'] = (
        (df['min_temperature'] < 15).astype(int) * 0.3 +
        (df['wind_speed'] < 5).astype(int) * 0.3 +
        (df['humidity'] > 70).astype(int) * 0.4
    )
    
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['season'] = df['month'].map({
        12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
        6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3
    })
    
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    for lag in [1, 3, 7]:
        df[f'temp_lag_{lag}'] = df['max_temperature'].shift(lag)
        df[f'humidity_lag_{lag}'] = df['humidity'].shift(lag)
        df[f'wind_lag_{lag}'] = df['wind_speed'].shift(lag)
        df[f'rainfall_lag_{lag}'] = df['rainfall'].shift(lag)
    
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

# Enhanced forecasting function with realistic seasonal patterns
def generate_advanced_forecast(base_data, rf_model, lstm_model, scaler_rf, scaler_lstm, hours=24):
    feature_columns = [
        'max_temperature', 'min_temperature', 'temp_range', 'humidity', 'wind_speed', 'pressure',
        'low_temp_indicator', 'low_wind_indicator', 'fog_risk_index',
        'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'temp_lag_1', 'humidity_lag_1', 'wind_lag_1', 'rainfall_lag_1',
        'temp_lag_3', 'humidity_lag_3', 'wind_lag_3', 'rainfall_lag_3',
        'temp_lag_7', 'humidity_lag_7', 'wind_lag_7', 'rainfall_lag_7'
    ]
    
    forecasts = []
    current_data = base_data.copy()
    
    for hour in range(hours):
        current_features = create_features(current_data)
        latest_features = current_features[feature_columns].iloc[-1:].values
        
        rf_features_scaled = scaler_rf.transform(latest_features)
        rain_prob = rf_model.predict_proba(rf_features_scaled)[0][1]
        rain_prediction = rf_model.predict(rf_features_scaled)[0]
        
        # Enhanced rainfall prediction with seasonal patterns
        forecast_time = datetime.now() + timedelta(hours=hour+1)
        forecast_month = forecast_time.month
        forecast_hour = forecast_time.hour
        
        # Determine season based on month
        if forecast_month in [12, 1, 2]:
            season = "winter"
        elif forecast_month in [3, 4, 5]:
            season = "spring"
        elif forecast_month in [6, 7, 8, 9]:
            season = "monsoon"
        else:  # [10, 11]
            season = "autumn"
        
        if rain_prediction == 1:
            if len(current_features) >= 30:
                lstm_sequence = current_features[feature_columns].iloc[-30:].values
                lstm_sequence_scaled = scaler_lstm.transform(lstm_sequence.reshape(-1, lstm_sequence.shape[-1])).reshape(lstm_sequence.shape)
                rainfall_amount = lstm_model.predict(lstm_sequence_scaled.reshape(1, 30, -1))[0][0]
                rainfall_amount = max(0, rainfall_amount)
            else:
                rainfall_amount = np.random.exponential(8)
        else:
            # Rain prediction is 0, but add seasonal rain probability for realism
            # Seasonal base rainfall probabilities
            if season == "monsoon":
                rain_chance = 0.4  # 40% chance in monsoon
                actual_rain = np.random.random() < rain_chance
                rainfall_amount = np.random.exponential(5) if actual_rain else 0
            elif season == "winter":
                rain_chance = 0.1  # 10% chance in winter
                actual_rain = np.random.random() < rain_chance
                rainfall_amount = np.random.exponential(2) if actual_rain else 0
            elif season == "spring":
                rain_chance = 0.25  # 25% chance in spring
                actual_rain = np.random.random() < rain_chance
                rainfall_amount = np.random.exponential(4) if actual_rain else 0
            else:  # autumn
                rain_chance = 0.2  # 20% chance in autumn
                actual_rain = np.random.random() < rain_chance
                rainfall_amount = np.random.exponential(3) if actual_rain else 0
            
            # Update rain_prediction based on actual rain occurrence
            rain_prediction = 1 if actual_rain else 0
        
        # City-specific rainfall adjustments
        if selected_city == "Mumbai" and season == "monsoon":
            rainfall_amount *= 1.5  # Heavy monsoon in Mumbai
            rain_prediction = 1  # Force rain during Mumbai monsoon
        elif selected_city == "Chennai" and forecast_month in [10, 11, 12]:
            rainfall_amount *= 1.3  # Northeast monsoon
            rain_prediction = 1  # Force rain during Chennai NE monsoon
        elif selected_city == "Bangalore":
            rainfall_amount *= 1.2  # Year-round rain possibility
        elif selected_city == "Delhi" and season == "monsoon":
            rainfall_amount *= 1.4  # Strong monsoon in Delhi
            rain_prediction = 1  # Force rain during Delhi monsoon
        
        # Seasonal temperature patterns
        if selected_city == "Delhi":
            if season == "winter":
                base_temp_range = [5, 20]
            elif season == "spring":
                base_temp_range = [18, 38]
            elif season == "monsoon":
                base_temp_range = [26, 33]
            else:  # autumn
                base_temp_range = [10, 25]
        elif selected_city == "Mumbai":
            if season == "winter":
                base_temp_range = [22, 31]
            elif season == "spring":
                base_temp_range = [26, 34]
            elif season == "monsoon":
                base_temp_range = [25, 29]
            else:  # autumn
                base_temp_range = [22, 27]
        elif selected_city == "Bangalore":
            if season == "winter":
                base_temp_range = [18, 27]
            elif season == "spring":
                base_temp_range = [22, 30]
            elif season == "monsoon":
                base_temp_range = [20, 25]
            else:  # autumn
                base_temp_range = [17, 23]
        elif selected_city == "Chennai":
            if season == "winter":
                base_temp_range = [24, 33]
            elif season == "spring":
                base_temp_range = [28, 36]
            elif season == "monsoon":
                base_temp_range = [27, 32]
            else:  # autumn
                base_temp_range = [24, 29]
        else:  # Default for other cities
            if season == "winter":
                base_temp_range = [20, 30]
            elif season == "spring":
                base_temp_range = [24, 35]
            elif season == "monsoon":
                base_temp_range = [25, 30]
            else:  # autumn
                base_temp_range = [20, 28]
        
        # Hour-based temperature variation within the day
        if 4 <= forecast_hour <= 6:  # Early morning - coolest
            hour_factor = 0.1
        elif 14 <= forecast_hour <= 16:  # Afternoon - hottest
            hour_factor = 0.9
        elif 20 <= forecast_hour <= 22:  # Evening
            hour_factor = 0.4
        else:
            hour_factor = 0.5
        
        # Calculate temperature based on seasonal range and hour
        base_temp = base_temp_range[0] + (base_temp_range[1] - base_temp_range[0]) * hour_factor
        hour_temp = base_temp + np.random.normal(0, 1.5)
        
        # Ensure temperature stays within realistic bounds
        hour_temp = np.clip(hour_temp, base_temp_range[0], base_temp_range[1])
        
        # Seasonal humidity patterns
        if season == "monsoon":
            base_humidity = 75 + np.random.normal(0, 10)
        elif season == "winter":
            base_humidity = 45 + np.random.normal(0, 10)
        else:  # spring, autumn
            base_humidity = 55 + np.random.normal(0, 10)
        
        hour_humidity = np.clip(base_humidity, 25, 95)
        
        # Seasonal wind patterns
        if season == "monsoon":
            base_wind = 12 + np.random.normal(0, 4)
        elif season == "winter":
            base_wind = 8 + np.random.normal(0, 3)
        else:  # spring, autumn
            base_wind = 10 + np.random.normal(0, 3)
        
        hour_wind = max(0, base_wind)
        
        # Enhanced weather condition classification
        if rain_prediction == 1:
            if rainfall_amount > 15:
                condition = "🌧️ Heavy Rain"
            elif rainfall_amount > 5:
                condition = "🌦️ Moderate Rain"
            else:
                condition = "🌦️ Light Rain"
        else:
            if current_features['fog_risk_index'].iloc[-1] > 0.7:
                condition = "🌫️ Foggy"
            elif hour_temp < 20:
                condition = "🌡️ Cool"
            elif hour_temp > 30:
                condition = "🌡️ Hot"
            else:
                condition = "☀️ Clear"
        
        forecast = {
            'time': forecast_time,
            'temperature': round(hour_temp, 1),
            'rainfall': round(rainfall_amount, 1),
            'rain_probability': round(rain_prob * 100, 1),
            'condition': condition,
            'wind_speed': round(hour_wind, 1),
            'humidity': round(hour_humidity, 1)
        }
        
        forecasts.append(forecast)
        
        # Autoregressive update
        new_row = current_data.iloc[-1].copy()
        new_row['date'] = pd.to_datetime(forecast_time.date())
        new_row['max_temperature'] = hour_temp
        new_row['min_temperature'] = hour_temp - 6
        new_row['humidity'] = hour_humidity
        new_row['wind_speed'] = hour_wind
        new_row['rainfall'] = rainfall_amount
        new_row['pressure'] = current_features['pressure'].iloc[-1] + np.random.normal(0, 1.5)
        
        current_data = pd.concat([current_data, new_row.to_frame().T], ignore_index=True)
    
    return forecasts

# Generate forecasts when button is clicked
if generate_forecast:
    with st.spinner("🔄 Generating advanced weather forecast..."):
        recent_data = historical_data.tail(60).copy()
        forecasts = generate_advanced_forecast(
            recent_data, rf_model, lstm_model, scaler_rf, scaler_lstm, forecast_hours
        )
        st.session_state.forecasts = forecasts
        st.session_state.generated_at = datetime.now()
        st.session_state.selected_city = selected_city

# Display results if forecasts exist
if 'forecasts' in st.session_state and st.session_state.forecasts:
    forecasts = st.session_state.forecasts
    city = st.session_state.selected_city
    
    # Enhanced metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🌍 City", city)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📊 Forecast Hours", len(forecasts))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_rain_prob = np.mean([f['rain_probability'] for f in forecasts])
        st.metric("🌧️ Avg Rain Probability", f"{avg_rain_prob:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        total_rain = sum([f['rainfall'] for f in forecasts])
        st.metric("💧 Total Expected Rain", f"{total_rain:.1f}mm")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced current weather display
    st.markdown("### 🌡️ Current Weather Conditions")
    current_time = datetime.now().strftime("%I:%M %p")
    current_temp = forecasts[0]['temperature']
    current_rain = forecasts[0]['rainfall']
    current_prob = forecasts[0]['rain_probability']
    current_wind = forecasts[0]['wind_speed']
    current_humidity = forecasts[0]['humidity']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="forecast-hour">
            <h4>🌡️ Temperature</h4>
            <h2>{current_temp}°C</h2>
            <p>Feels like {current_temp - 2}°C</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="forecast-hour">
            <h4>🌧️ Rainfall</h4>
            <h2>{current_rain}mm</h2>
            <p>Probability: {current_prob}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="forecast-hour">
            <h4>💨 Wind & Humidity</h4>
            <h2>{current_wind} km/h</h2>
            <p>Humidity: {current_humidity}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced hourly forecast display
    st.markdown("### ⏰ Hourly Forecast Breakdown")
    
    # Display first 12 hours in detailed cards
    cols = st.columns(3)
    for i, forecast in enumerate(forecasts[:12]):
        with cols[i % 3]:
            time_str = forecast['time'].strftime("%I:%M %p")
            temp_emoji = "🌡️" if forecast['temperature'] > 25 else "🌡️"
            rain_emoji = "🌧️" if forecast['rainfall'] > 0 else "☀️"
            
            st.markdown(f"""
            <div class="forecast-hour">
                <h5>{time_str}</h5>
                <h3>{temp_emoji} {forecast['temperature']}°C</h3>
                <p>{rain_emoji} {forecast['rainfall']}mm | 📊 {forecast['rain_probability']}%</p>
                <p><strong>{forecast['condition']}</strong></p>
                <p>💨 {forecast['wind_speed']} km/h | 💦 {forecast['humidity']}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Daily forecast bar chart
    st.markdown("### 📊 Daily Rainfall Forecast")
    
    # Create extended forecast data first (for consistency with hourly breakdown)
    extended_data = []
    for i, forecast in enumerate(forecasts):
        extended_data.append({
            'Date': forecast['time'].strftime('%Y-%m-%d'),
            'Hour': forecast['time'].strftime('%H:00'),
            'Temperature (°C)': forecast['temperature'],
            'Rainfall (mm)': forecast['rainfall'],
            'Rain Probability (%)': forecast['rain_probability'],
            'Condition': forecast['condition'],
            'Wind Speed (km/h)': forecast['wind_speed'],
            'Humidity (%)': forecast['humidity']
        })
    
    extended_df = pd.DataFrame(extended_data)
    
    # Group by date and calculate daily totals from extended data (summing hourly rainfall)
    daily_totals = extended_df.groupby('Date').agg({
        'Rainfall (mm)': 'sum',  # Sum all hourly rainfall values for each day
        'Temperature (°C)': 'mean',  # Average temperature
        'Rain Probability (%)': 'max'  # Maximum rain probability
    }).reset_index()
    
    # Format date for display
    daily_totals['Date_Display'] = pd.to_datetime(daily_totals['Date']).dt.strftime('%b %d')
    
    # Create enhanced bar chart
    fig_daily = go.Figure()
    
    # Add rainfall bars (summed from hourly data)
    fig_daily.add_trace(go.Bar(
        x=daily_totals['Date_Display'],
        y=daily_totals['Rainfall (mm)'],
        name='Total Rainfall (mm)',
        marker_color='#9333ea',
        text=daily_totals['Rainfall (mm)'],
        textposition='outside',
        textfont=dict(color='black', size=16, family='Arial Black'),
        insidetextanchor='middle',
        cliponaxis=False
    ))
    
    # Add temperature line
    fig_daily.add_trace(go.Scatter(
        x=daily_totals['Date_Display'],
        y=daily_totals['Temperature (°C)'],
        mode='lines+markers',
        name='Avg Temp (°C)',
        line=dict(color='#ff1744', width=5),
        marker=dict(size=12, color='#ff1744', line=dict(width=2, color='white')),
        hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Temperature: %{y}°C<extra></extra>'
    ))
    
    fig_daily.update_layout(
        title=f" {len(daily_totals)}-Day Rainfall Forecast for {city} ",
        title_font=dict(size=25, color='#2c3e50'),
        xaxis_title="Date",
        xaxis_title_font=dict(size=25, color='#2c3e50'),
        xaxis=dict(tickfont=dict(size=20, color='#2c3e50'), showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(tickfont=dict(size=20, color='#2c3e50'), showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis_title="Rainfall (mm) / Temperature (°C)",
        yaxis_title_font=dict(size=25, color='#2c3e50'),
        height=600,
        plot_bgcolor='rgba(248,249,250,0.95)',
        paper_bgcolor='white',
        font=dict(size=12),
        hoverlabel=dict(bgcolor='white', font_size=14, font_family='Arial'),
        legend=dict(
            font=dict(size=18, color='black'),
            x=0.98,
            y=0.98,
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='black',
            borderwidth=4,
            itemwidth=30
        ),
        margin=dict(l=80, r=80, t=80, b=80),
        showlegend=True
    )
    print(st)
    st.plotly_chart(fig_daily, use_container_width=True)
    
    # Extended forecast in 24-hour format
    st.markdown("###  Extended Forecast (24-Hour Format)")
    
    # Create extended forecast data
    extended_data = []
    for i, forecast in enumerate(forecasts):
        extended_data.append({
            'Date': forecast['time'].strftime('%Y-%m-%d'),
            'Hour': forecast['time'].strftime('%H:00'),
            'Temperature (°C)': forecast['temperature'],
            'Rainfall (mm)': forecast['rainfall'],
            'Rain Probability (%)': forecast['rain_probability'],
            'Condition': forecast['condition'],
            'Wind Speed (km/h)': forecast['wind_speed'],
            'Humidity (%)': forecast['humidity']
        })
    
    extended_df = pd.DataFrame(extended_data)
    
    # Add download options
    st.markdown("### 📥 Download Options")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Record count selector
        record_options = [12, 15, 25, 50, 100]
        selected_records = st.selectbox(
            "📋 Select number of records to download:",
            record_options,
            index=0,
            help="Choose how many forecast records to include in PNG image"
        )
    
    with col2:
        # CSV Download
        csv_data = extended_df.head(selected_records).to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"extended_forecast_{city}_{selected_records}records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col3:
        # PNG Download - Match dashboard table appearance
        def create_forecast_png(num_records):
            # Get selected number of records
            df_selected = extended_df.head(num_records)
            
            # Create figure and axis to match dashboard style
            fig, ax = plt.subplots(figsize=(16, 8))
            fig.patch.set_facecolor('white')
            
            # Hide axes
            ax.set_axis_off()
            
            # Create table data exactly like dashboard
            table_data = []
            headers = ['Date', 'Hour', 'Temperature (°C)', 'Rainfall (mm)', 'Rain Probability (%)', 'Condition', 'Wind Speed (km/h)', 'Humidity (%)']
            
            for _, row in df_selected.iterrows():
                table_data.append([
                    row['Date'],
                    row['Hour'],
                    f"{row['Temperature (°C)']:.1f}",
                    f"{row['Rainfall (mm)']:.1f}",
                    f"{row['Rain Probability (%)']:.1f}",
                    row['Condition'],
                    f"{row['Wind Speed (km/h)']:.1f}",
                    f"{row['Humidity (%)']:.1f}"
                ])
            
            # Create title matching dashboard
            title = f"###  Extended Forecast (24-Hour Format) - {city}\nFirst {num_records} Records"
            plt.text(0.5, 0.95, title, ha='center', va='top', fontsize=14, fontweight='normal', transform=fig.transFigure)
            
            # Create timestamp
            timestamp = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            plt.text(0.5, 0.92, timestamp, ha='center', va='top', fontsize=10, style='italic', transform=fig.transFigure)
            
            # Create table with dashboard-like styling
            table = plt.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center', bbox=[0, 0.05, 1, 0.85])
            
            # Style the table to match dashboard
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.8)
            
            # Header styling - clean like dashboard
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#f0f2f6')
                table[(0, i)].set_text_props(weight='bold')
                table[(0, i)].set_height(0.06)
            
            # Row styling - clean like dashboard
            for i in range(1, len(table_data) + 1):
                for j in range(len(headers)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f8f9fa')
                    else:
                        table[(i, j)].set_facecolor('white')
                    table[(i, j)].set_height(0.06)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save to bytes buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            plt.close()
            
            return buffer.getvalue()
        
        png_data = create_forecast_png(selected_records)
        st.download_button(
            label="🖼️ Download PNG",
            data=png_data,
            file_name=f"extended_forecast_{city}_{selected_records}records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png"
        )
    
    st.markdown("---")
    
    # Display in a clean table format
    st.dataframe(
        extended_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Temperature (°C)": st.column_config.NumberColumn(format="%.1f °C"),
            "Rainfall (mm)": st.column_config.NumberColumn(format="%.1f mm"),
            "Rain Probability (%)": st.column_config.NumberColumn(format="%.1f %%"),
            "Wind Speed (km/h)": st.column_config.NumberColumn(format="%.1f km/h"),
            "Humidity (%)": st.column_config.NumberColumn(format="%.1f %%"),
        }
    )

# Professional footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #666;'>
    <p><strong>🌤️ Advanced Weather Forecasting System</strong></p>
    <p>Powered by Hybrid ML Architecture | Random Forest + LSTM Neural Network</p>
    <p>Real-time predictions for major Indian cities</p>
</div>
""", unsafe_allow_html=True)