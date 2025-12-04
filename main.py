import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

from prophet import Prophet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split                                                               
# OpenWeatherMap API key
OPENWEATHER_API_KEY = "YOUR_API_KEY"   # <-- put your key here

# =========================
# CONFIG FOR YOUR DATASET
# =========================
DATA_PATH = "data/india_2000_2024_daily_weather.csv"
CITY_NAME = "Delhi"   # you can change this later
DATE_COL = "date"
CITY_COL = "city"
TMAX_COL = "temperature_2m_max"
TMIN_COL = "temperature_2m_min"
RAIN_COL = "precipitation_sum"      # we'll use this to define "rain"
WIND_COL = "wind_speed_10m_max"

# =========================
# 1. LOAD & CLEAN DATA
# =========================
def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)

    # Parse date
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # Filter city
    if CITY_COL in df.columns:
        df = df[df[CITY_COL].str.lower() == CITY_NAME.lower()]

    # Sort by date and drop missing important cols
    important_cols = [TMAX_COL, TMIN_COL, RAIN_COL]
    df = df.sort_values(DATE_COL)
    df = df.dropna(subset=important_cols)

    df = df.reset_index(drop=True)
    print("Data shape after cleaning:", df.shape)
    print(df.head())
    return df


# =========================
# 2. TEMP FORECAST (PROPHET)
# =========================
def train_prophet_temperature(df):
    """
    Train Prophet on daily max temperature and forecast next 7 days.
    """

    temp_df = df[[DATE_COL, TMAX_COL]].rename(columns={DATE_COL: "ds", TMAX_COL: "y"})

    # Train / test split (last 30 days as test if enough data)
    if len(temp_df) <= 60:
        split_idx = int(len(temp_df) * 0.8)
    else:
        split_idx = -30

    train = temp_df.iloc[:split_idx]
    test = temp_df.iloc[split_idx:]

    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(train)

    # Forecast through test period + 7 future days
    days_to_forecast = len(test) + 7
    future = model.make_future_dataframe(periods=days_to_forecast)
    forecast = model.predict(future)

    # Evaluate on test
    forecast_test = forecast.set_index("ds").loc[test["ds"]]
    mae = mean_absolute_error(test["y"].values, forecast_test["yhat"].values)
    print(f"\n[Prophet] Temperature MAE on last {len(test)} days: {mae:.3f} °C")

    # Next 7-day forecast
    future_7 = forecast.tail(7)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    print("\nNext 7-day temperature forecast:")
    print(future_7)

    # Plots
    model.plot(forecast)
    plt.title(f"{CITY_NAME} Temperature Forecast (Prophet)")
    plt.tight_layout()
    plt.show()

    model.plot_components(forecast)
    plt.tight_layout()
    plt.show()

    return model, future_7


# =========================
# 3. RAIN PROBABILITY MODEL
# =========================
def build_rain_labels(df, rain_threshold=0.1):
    """
    Define rain if precipitation_sum > threshold.
    Create label 'rain_tomorrow' for next day.
    """
    df = df.copy()
    df["rain_today"] = df[RAIN_COL] > rain_threshold
    df["rain_tomorrow"] = df["rain_today"].shift(-1)
    df = df.dropna(subset=["rain_tomorrow"])
    return df


def create_rain_features(df):
    """
    Create lag features for temperature, rain and wind.
    """
    df = df.copy()

    for lag in [1, 2, 3]:
        df[f"{TMAX_COL}_lag{lag}"] = df[TMAX_COL].shift(lag)
        df[f"{TMIN_COL}_lag{lag}"] = df[TMIN_COL].shift(lag)
        df[f"{RAIN_COL}_lag{lag}"] = df[RAIN_COL].shift(lag)
        df[f"{WIND_COL}_lag{lag}"] = df[WIND_COL].shift(lag)

    df = df.dropna()

    feature_cols = [c for c in df.columns if "lag" in c]
    X = df[feature_cols]
    y = df["rain_tomorrow"].astype(int)

    return X, y, feature_cols, df


def train_rain_model(df):
    """
    Logistic Regression model that outputs probability of rain tomorrow.
    """
    df_labels = build_rain_labels(df)
    X, y, feature_cols, df_final = create_rain_features(df_labels)

    # Time-series split (no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n[Rain Model] Accuracy on last {len(y_test)} days: {acc:.3f}")

    return clf, feature_cols, df_final


def get_latest_rain_features(df_final, feature_cols):
    latest_row = df_final.iloc[-1]
    return latest_row[feature_cols].values.reshape(1, -1)


# =========================
# 4. OPENWEATHERMAP (LIVE DATA)
# =========================
def fetch_today_weather(city_name):
    # ✅ Only skip if key is truly missing
    if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY.strip() == "":
        print("\n[Live] Skipping API call (no API key set).")
        return None

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": f"{city_name},IN",
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)

        if resp.status_code != 200:
            print("\n[Live] API Error:", resp.json())
            return None

        data = resp.json()

        desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind = data["wind"]["speed"]

        print(f"\n[Live] {city_name}: {desc}, {temp}°C, humidity {humidity}%, wind {wind} m/s")
        return data

    except Exception as e:
        print("\n[Live] API Connection Error:", e)
        return None


# =========================
# 5. WEATHER INTELLIGENCE AGENT
# =========================
def weather_intelligence_agent(city_name, temp_forecast_7d, rain_model, df_final, feature_cols):
    print("\n==============================")
    print(f"  Weather Intelligence Agent")
    print(f"           {city_name}")
    print("==============================")

    # 1) Live weather today
    try:
        fetch_today_weather(city_name)
    except Exception as e:
        print("Could not fetch live weather (check API key / internet):", e)

    # 2) Next 7-day temperature summary
    print("\nTemperature forecast for next 7 days (°C):")
    for _, row in temp_forecast_7d.iterrows():
        date_str = row["ds"].date().isoformat()
        print(
            f"{date_str}: {row['yhat']:.1f} (range {row['yhat_lower']:.1f} – {row['yhat_upper']:.1f})"
        )

    # 3) Probability of rain tomorrow
    X_latest = get_latest_rain_features(df_final, feature_cols)
    rain_prob = rain_model.predict_proba(X_latest)[0, 1]

    print(f"\nEstimated chance of rain tomorrow: {rain_prob*100:.1f}%")

    # 4) Recommendations
    if rain_prob >= 0.7:
        print("💡 Recommendation: High chance of rain. Definitely carry an umbrella ☔")
    elif rain_prob >= 0.4:
        print("💡 Recommendation: Moderate chance of rain. Better to keep an umbrella just in case.")
    else:
        print("💡 Recommendation: Low chance of rain. Umbrella not required, but check forecast if traveling.")

    mean_temp = temp_forecast_7d["yhat"].mean()
    if mean_temp >= 35:
        print("🔥 It will be quite hot on average. Stay hydrated and avoid peak afternoon heat.")
    elif mean_temp <= 20:
        print("🥶 Cooler weather ahead. You might need a light jacket in the evenings.")


# =========================
# 6. MAIN
# =========================
def main():
    df = load_and_prepare_data()
    temp_model, temp_forecast_7d = train_prophet_temperature(df)
    rain_model, feature_cols, df_final = train_rain_model(df)
    weather_intelligence_agent(CITY_NAME, temp_forecast_7d, rain_model, df_final, feature_cols)


if __name__ == "__main__":
    main()

