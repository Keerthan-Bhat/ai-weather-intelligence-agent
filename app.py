import streamlit as st
import pandas as pd
import numpy as np
import requests
import os  #add this

from prophet import Prophet
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
DATA_PATH = "data/india_2000_2024_daily_weather.csv"
DEFAULT_CITY = "Delhi"   # for now we focus on Delhi

# Read API key from environment variable (SAFE)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")



# =========================
# DATA & MODELS
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


@st.cache_resource
def train_models(df, city_name):
    # Filter city
    city_df = df[df["city"].str.lower() == city_name.lower()].copy()
    city_df = city_df.dropna(
        subset=[
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
        ]
    )

    # ===== Prophet temperature model =====
    temp_df = city_df[["date", "temperature_2m_max"]].rename(
        columns={"date": "ds", "temperature_2m_max": "y"}
    )

    if len(temp_df) <= 60:
        split_idx = int(len(temp_df) * 0.8)
    else:
        split_idx = -30

    train = temp_df.iloc[:split_idx]
    test = temp_df.iloc[split_idx:]

    temp_model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    temp_model.fit(train)

    # 1) Forecast enough to cover TEST period for MAE
    future_for_test = temp_model.make_future_dataframe(periods=len(test))
    forecast_for_test = temp_model.predict(future_for_test)

    # Align by date, inner join to avoid KeyError
    forecast_idx = forecast_for_test.set_index("ds")
    test_idx = test.set_index("ds")
    aligned = test_idx.join(forecast_idx[["yhat"]], how="inner")

    mae = float(np.mean(np.abs(aligned["y"] - aligned["yhat"])))

    # 2) Forecast full + 7 future days for display
    future_full = temp_model.make_future_dataframe(periods=7)
    forecast_full = temp_model.predict(future_full)

    forecast_7 = forecast_full.tail(7)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    # ===== Rain probability model =====
    df_labels = city_df.copy()
    df_labels["rain_today"] = df_labels["precipitation_sum"] > 0.1
    df_labels["rain_tomorrow"] = df_labels["rain_today"].shift(-1)
    df_labels = df_labels.dropna(subset=["rain_tomorrow"])

    for lag in [1, 2, 3]:
        df_labels[f"temperature_2m_max_lag{lag}"] = df_labels["temperature_2m_max"].shift(lag)
        df_labels[f"temperature_2m_min_lag{lag}"] = df_labels["temperature_2m_min"].shift(lag)
        df_labels[f"precipitation_sum_lag{lag}"] = df_labels["precipitation_sum"].shift(lag)
        df_labels[f"wind_speed_10m_max_lag{lag}"] = df_labels["wind_speed_10m_max"].shift(lag)

    df_labels = df_labels.dropna()
    feature_cols = [c for c in df_labels.columns if "lag" in c]

    X = df_labels[feature_cols]
    y = df_labels["rain_tomorrow"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    df_final = df_labels

    return {
        "city_df": city_df,
        "temp_model": temp_model,
        "temp_mae": mae,
        "forecast_full": forecast_full,
        "forecast_7": forecast_7,
        "rain_model": clf,
        "rain_features": feature_cols,
        "rain_df_final": df_final,
        "rain_acc": acc,
    }


def get_latest_rain_features(df_final, feature_cols):
    latest_row = df_final.iloc[-1]
    return latest_row[feature_cols].values.reshape(1, -1)


def fetch_live_weather(city_name):
    # Only disable if key is actually missing
    if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY.strip() == "":
        return None, "No API key set. Live weather disabled."


    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": f"{city_name},IN",
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return None, f"API error: {resp.json()}"

        data = resp.json()
        live = {
            "description": data["weather"][0]["description"],
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "wind": data["wind"]["speed"],
        }
        return live, None
    except Exception as e:
        return None, f"Connection error: {e}"


# =========================
# STREAMLIT UI
# =========================
def main():
    st.set_page_config(
        page_title="AI Weather Intelligence Agent",
        page_icon="🌦️",
        layout="wide",
    )

    st.title("🌦️ AI Weather Intelligence Agent")
    st.caption("Forecasts, rain probability, and smart recommendations using ML + real API data.")

    df = load_data()
    cities = sorted(df["city"].unique())

    with st.sidebar:
        st.header("Settings")
        city = st.selectbox("Select City", options=cities, index=cities.index(DEFAULT_CITY) if DEFAULT_CITY in cities else 0)
        st.markdown("---")
        st.write("**Model Info**")
        st.write("- Prophet for temperature forecast")
        st.write("- Logistic Regression for rain probability")

    st.subheader(f"📍 City: {city}")

    models = train_models(df, city)

    # Live weather
    live_col, metrics_col = st.columns(2)

    with live_col:
        st.markdown("### 🌍 Live Weather (OpenWeatherMap)")
        live, err = fetch_live_weather(city)
        if err:
            st.warning(err)
        elif live:
            st.metric("Current Temp (°C)", f"{live['temp']:.1f}")
            st.metric("Humidity (%)", f"{live['humidity']:.0f}")
            st.metric("Wind (m/s)", f"{live['wind']:.2f}")
            st.write(f"**Conditions:** {live['description'].title()}")
        else:
            st.info("Live weather unavailable.")

    # Forecast metrics
    with metrics_col:
        st.markdown("### 📈 7-Day Temperature Forecast")
        f7 = models["forecast_7"]
        avg_temp = f7["yhat"].mean()
        st.metric("Avg forecast (°C)", f"{avg_temp:.1f}")
        st.metric("Temp MAE (last 30 days)", f"{models['temp_mae']:.2f} °C")
        st.metric("Rain model accuracy", f"{models['rain_acc']*100:.1f}%")

    # Charts
    st.markdown("### 📊 Temperature History & Forecast")

    city_df = models["city_df"]
    forecast_full = models["forecast_full"]

    hist = city_df[["date", "temperature_2m_max"]].rename(columns={"date": "ds", "temperature_2m_max": "y"})
    hist = hist.set_index("ds").tail(365)  # last year

    future_plot = forecast_full.set_index("ds")[["yhat"]].tail(7)
    combined = hist.join(future_plot, how="outer", rsuffix="_forecast")

    st.line_chart(combined)

    # Rain probability + recommendation
    st.markdown("### 🌧️ Rain Probability & Recommendation")

    rain_model = models["rain_model"]
    feature_cols = models["rain_features"]
    df_final = models["rain_df_final"]

    X_latest = get_latest_rain_features(df_final, feature_cols)
    rain_prob = rain_model.predict_proba(X_latest)[0, 1]

    st.write(f"**Estimated chance of rain tomorrow:** `{rain_prob*100:.1f}%`")

    if rain_prob >= 0.7:
        st.success("High chance of rain. Definitely carry an umbrella ☔")
    elif rain_prob >= 0.4:
        st.warning("Moderate chance of rain. Better to keep an umbrella just in case.")
    else:
        st.info("Low chance of rain. Umbrella not required, but check latest forecast if traveling.")

    if avg_temp >= 35:
        st.warning("🔥 It will be quite hot on average. Stay hydrated and avoid peak afternoon heat.")
    elif avg_temp <= 20:
        st.info("🥶 Cooler weather ahead. You might need a light jacket in the evenings.")


if __name__ == "__main__":
    main()
