# 🌦️ AI Weather Intelligence Agent

A complete end-to-end machine learning project that performs:

- 📈 7-day temperature forecasting using **Prophet**
- 🌧️ Rain probability prediction using **Logistic Regression**
- 🌍 Real-time weather data using **OpenWeatherMap API**
- 🧠 Smart recommendations like umbrella alerts & cold-weather warnings
- 🖥️ Interactive **Streamlit Dashboard**

This project uses **25+ years of historical Indian weather data** and combines **time-series forecasting, classification, and live API integration** into a single intelligent system.

---

## 🚀 Key Features

✅ 7-day temperature forecast  
✅ Rain probability prediction (81% accuracy)  
✅ Real-time live weather  
✅ Smart recommendations (umbrella, heat, cold alerts)  
✅ Interactive Streamlit web dashboard  
✅ Safe API key handling using environment variables  

---

## 🧠 Tech Stack

- **Language:** Python
- **Libraries:**
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - prophet
  - requests
  - streamlit
- **API:** OpenWeatherMap

---

## 📁 Project Structure

```bash
weather-agent/
├── data/
│   └── india_2000_2024_daily_weather.csv
├── app.py          # Streamlit dashboard
├── main.py         # CLI ML pipeline
├── requirements.txt
├── README.md
├── .gitignore
└── .venv/          # Virtual environment (not uploaded)
