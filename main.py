import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np

def preprocess_data():
    df = pd.read_csv('weatherHistory.csv')

    # Convert date
    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
    df['month'] = df['Formatted Date'].dt.month
    df['day'] = df['Formatted Date'].dt.day
    df['hour'] = df['Formatted Date'].dt.hour

    # Simplify columns
    df.rename(columns={
        'Temperature (C)': 'temperature',
        'Humidity': 'humidity',
        'Wind Speed (km/h)': 'wind_speed',
        'Pressure (millibars)': 'pressure',
        'Summary': 'summary'
    }, inplace=True)

    df.fillna(method='ffill', inplace=True)
    return df

def train_models():
    df = preprocess_data()
    features = ['month', 'day', 'hour']

    X = df[features]

    # 1. Temperature Model
    y_temp = df['temperature']
    temp_model = RandomForestRegressor(n_estimators=100, random_state=42)
    temp_model.fit(X, y_temp)

    # 2. Humidity Model
    y_humidity = df['humidity']
    humidity_model = RandomForestRegressor(n_estimators=100, random_state=42)
    humidity_model.fit(X, y_humidity)

    # 3. Summary Model (Classification)
    le = LabelEncoder()
    df['summary_encoded'] = le.fit_transform(df['summary'])
    y_summary = df['summary_encoded']

    summary_model = RandomForestClassifier(n_estimators=100, random_state=42)
    summary_model.fit(X, y_summary)

    # Save models
    with open('temp_model.pkl', 'wb') as f:
        pickle.dump(temp_model, f)

    with open('humidity_model.pkl', 'wb') as f:
        pickle.dump(humidity_model, f)

    with open('summary_model.pkl', 'wb') as f:
        pickle.dump(summary_model, f)

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print("Models trained and saved!")

def predict_weather(month, day, hour):
    # Load models
    with open('temp_model.pkl', 'rb') as f:
        temp_model = pickle.load(f)
    with open('humidity_model.pkl', 'rb') as f:
        humidity_model = pickle.load(f)
    with open('summary_model.pkl', 'rb') as f:
        summary_model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    # Prepare input
    features = np.array([[month, day, hour]])

    # Predict
    temp = temp_model.predict(features)[0]
    humidity = humidity_model.predict(features)[0]
    summary_encoded = summary_model.predict(features)[0]
    summary = le.inverse_transform([summary_encoded])[0]

    return round(temp, 2), round(humidity, 2), summary

# Optional: Run training
if __name__ == "__main__":
    train_models()
    print(predict_weather(3, 15, 14))  # Example usage
