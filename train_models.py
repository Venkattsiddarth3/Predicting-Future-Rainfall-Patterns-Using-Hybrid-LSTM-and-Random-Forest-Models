import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("PHASE 1: DATA LOADING & PREPROCESSING")
print("=" * 60)

# Load dataset
print("Loading dataset...")
df = pd.read_csv('india_2000_2024_daily_weather.csv')
print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Convert date to datetime and sort
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Handle missing values
print(f"Missing values before handling:\n{df.isnull().sum()}")
df = df.fillna(method='ffill').fillna(method='bfill')
print(f"Missing values after handling:\n{df.isnull().sum()}")

print("\n" + "=" * 60)
print("PHASE 1: FEATURE ENGINEERING")
print("=" * 60)

def create_features(df, city='Delhi'):
    """Create Delhi-specific weather features"""
    df = df.copy()
    
    # Temperature range
    df['temp_range'] = df['max_temperature'] - df['min_temperature']
    
    # Low temperature indicator (below 15°C)
    df['low_temp_indicator'] = (df['min_temperature'] < 15).astype(int)
    
    # Low wind indicator (below 5 km/h)
    df['low_wind_indicator'] = (df['wind_speed'] < 5).astype(int)
    
    # Fog risk index (combination of low temp, low wind, high humidity)
    df['fog_risk_index'] = (
        (df['min_temperature'] < 15).astype(int) * 0.3 +
        (df['wind_speed'] < 5).astype(int) * 0.3 +
        (df['humidity'] > 70).astype(int) * 0.4
    )
    
    # Seasonal encoding
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['season'] = df['month'].map({
        12: 0, 1: 0, 2: 0,  # Winter
        3: 1, 4: 1, 5: 1,   # Spring/Pre-monsoon
        6: 2, 7: 2, 8: 2,   # Monsoon
        9: 3, 10: 3, 11: 3  # Autumn/Post-monsoon
    })
    
    # Cyclical encoding for seasonality
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Lag features (1, 3, 7 days)
    for lag in [1, 3, 7]:
        df[f'temp_lag_{lag}'] = df['max_temperature'].shift(lag)
        df[f'humidity_lag_{lag}'] = df['humidity'].shift(lag)
        df[f'wind_lag_{lag}'] = df['wind_speed'].shift(lag)
        df[f'rainfall_lag_{lag}'] = df['rainfall'].shift(lag)
    
    # Rain occurrence target for Random Forest
    df['rain_occurrence'] = (df['rainfall'] > 0).astype(int)
    
    # Fill NaN values created by lag features
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

# Create features
print("Creating Delhi-specific features...")
df_features = create_features(df)
print(f"Features created. New shape: {df_features.shape}")

# Define feature lists
feature_columns = [
    'max_temperature', 'min_temperature', 'temp_range', 'humidity', 'wind_speed', 'pressure',
    'low_temp_indicator', 'low_wind_indicator', 'fog_risk_index',
    'day_sin', 'day_cos', 'month_sin', 'month_cos',
    'temp_lag_1', 'humidity_lag_1', 'wind_lag_1', 'rainfall_lag_1',
    'temp_lag_3', 'humidity_lag_3', 'wind_lag_3', 'rainfall_lag_3',
    'temp_lag_7', 'humidity_lag_7', 'wind_lag_7', 'rainfall_lag_7'
]

print(f"Using {len(feature_columns)} features for modeling")

print("\n" + "=" * 60)
print("PHASE 2: MODEL TRAINING")
print("=" * 60)

# Split data (80% train, 20% test)
train_size = int(len(df_features) * 0.8)
train_data = df_features.iloc[:train_size]
test_data = df_features.iloc[train_size:]

print(f"Training data: {len(train_data)} samples")
print(f"Test data: {len(test_data)} samples")

# Prepare data for Random Forest
X_train_rf = train_data[feature_columns]
y_train_rf = train_data['rain_occurrence']
X_test_rf = test_data[feature_columns]
y_test_rf = test_data['rain_occurrence']

# Scale features for Random Forest
scaler_rf = StandardScaler()
X_train_rf_scaled = scaler_rf.fit_transform(X_train_rf)
X_test_rf_scaled = scaler_rf.transform(X_test_rf)

print("\nTraining Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_rf_scaled, y_train_rf)

# Evaluate Random Forest
rf_train_pred = rf_model.predict(X_train_rf_scaled)
rf_test_pred = rf_model.predict(X_test_rf_scaled)
rf_test_proba = rf_model.predict_proba(X_test_rf_scaled)[:, 1]

rf_train_acc = accuracy_score(y_train_rf, rf_train_pred)
rf_test_acc = accuracy_score(y_test_rf, rf_test_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test_rf, rf_test_proba))

print(f"Random Forest - Train Accuracy: {rf_train_acc:.4f}")
print(f"Random Forest - Test Accuracy: {rf_test_acc:.4f}")
print(f"Random Forest - RMSE: {rf_rmse:.4f}")

# Prepare data for LSTM
print("\nTraining LSTM Regression Model...")

# Create sequences for LSTM
def create_sequences(data, sequence_length=30):
    """Create sequences for LSTM training"""
    sequences = []
    targets = []
    
    # Only use sequences where it rained at the end
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length][feature_columns].values
        target = data.iloc[i + sequence_length]['rainfall']
        
        # Only include sequences ending with rainfall > 0
        if target > 0:
            sequences.append(seq)
            targets.append(target)
    
    return np.array(sequences), np.array(targets)

sequence_length = 30
X_lstm, y_lstm = create_sequences(train_data, sequence_length)

print(f"LSTM sequences created: {len(X_lstm)} sequences")
print(f"Sequence shape: {X_lstm.shape}")

# Scale LSTM data
scaler_lstm = StandardScaler()
X_lstm_scaled = scaler_lstm.fit_transform(X_lstm.reshape(-1, X_lstm.shape[-1])).reshape(X_lstm.shape)

# Split LSTM data
X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm = train_test_split(
    X_lstm_scaled, y_lstm, test_size=0.2, random_state=42
)

# Build LSTM model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, len(feature_columns))),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1, activation='linear')
])

lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train LSTM with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("Training LSTM...")
history = lstm_model.fit(
    X_train_lstm, y_train_lstm,
    validation_data=(X_val_lstm, y_val_lstm),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate LSTM
lstm_pred = lstm_model.predict(X_val_lstm)
lstm_rmse = np.sqrt(mean_squared_error(y_val_lstm, lstm_pred))

print(f"LSTM - Validation RMSE: {lstm_rmse:.4f}")

print("\n" + "=" * 60)
print("PHASE 3: SAVING TRAINED ARTIFACTS")
print("=" * 60)

# Save models and artifacts
print("Saving models and artifacts...")

# Save LSTM model
lstm_model.save('lstm_model.h5')
print("+ lstm_model.h5 saved")

# Save Random Forest model
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("+ rf_model.pkl saved")

# Save scalers
with open('scaler_lstm.pkl', 'wb') as f:
    pickle.dump(scaler_lstm, f)
with open('scaler_rf.pkl', 'wb') as f:
    pickle.dump(scaler_rf, f)
print("+ scaler_lstm.pkl and scaler_rf.pkl saved")

# Save metrics
metrics = {
    'random_forest': {
        'train_accuracy': float(rf_train_acc),
        'test_accuracy': float(rf_test_acc),
        'rmse': float(rf_rmse)
    },
    'lstm': {
        'validation_rmse': float(lstm_rmse),
        'sequence_length': sequence_length,
        'training_sequences': len(X_train_lstm),
        'validation_sequences': len(X_val_lstm)
    },
    'data_info': {
        'total_samples': len(df_features),
        'train_samples': len(train_data),
        'test_samples': len(test_data),
        'feature_count': len(feature_columns),
        'date_range': {
            'start': df_features['date'].min().strftime('%Y-%m-%d'),
            'end': df_features['date'].max().strftime('%Y-%m-%d')
        }
    }
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("+ metrics.json saved")

# Save sample predictions
print("\nGenerating sample predictions...")
sample_indices = np.random.choice(len(X_test_rf), min(10, len(X_test_rf)), replace=False)
sample_predictions = []

for i in sample_indices:
    actual_date = test_data.iloc[i]['date'].strftime('%Y-%m-%d')
    actual_rain = test_data.iloc[i]['rainfall']
    actual_rain_occurrence = test_data.iloc[i]['rain_occurrence']
    
    # Random Forest prediction
    rf_prob = rf_test_proba[i]
    rf_pred = rf_test_pred[i]
    
    sample_predictions.append({
        'date': actual_date,
        'actual_rainfall_mm': float(actual_rain),
        'actual_rain_occurrence': int(actual_rain_occurrence),
        'rf_rain_probability': float(rf_prob),
        'rf_rain_prediction': int(rf_pred)
    })

sample_df = pd.DataFrame(sample_predictions)
sample_df.to_csv('sample_predictions.csv', index=False)
print("+ sample_predictions.csv saved")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print("Files created:")
print("- lstm_model.h5 (LSTM regression model)")
print("- rf_model.pkl (Random Forest classification model)")
print("- scaler_lstm.pkl (LSTM feature scaler)")
print("- scaler_rf.pkl (Random Forest feature scaler)")
print("- metrics.json (Training metrics)")
print("- sample_predictions.csv (Sample predictions)")
print("\nModels are ready for use in Streamlit dashboard!")
