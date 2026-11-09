# ==========================================
# WEATHER DATA + WEATHER PREDICTION
# ==========================================

import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression


# ==========================================
# FIND CITY COORDINATES
# ==========================================

def get_city_coordinates(city: str) -> tuple[float, float]:
    """Find latitude and longitude for a city using Open-Meteo."""

    geocode_url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    response = requests.get(
        geocode_url,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    data = response.json().get("results")

    if not data:
        raise ValueError(
            f"Could not find coordinates for city: {city}"
        )

    location = data[0]

    return (
        float(location["latitude"]),
        float(location["longitude"])
    )


# ==========================================
# GET HISTORICAL WEATHER
# ==========================================

def fetch_historical_weather(
    city: str,
    years: int = 5,
    output_csv: str = "city_historical_weather.csv"
) -> pd.DataFrame:
    """
    Download historical daily weather data
    and save it to a CSV file.
    """

    lat, lon = get_city_coordinates(city)

    end_date = date.today()

    start_date = end_date - timedelta(
        days=int(years * 365.25)
    )

    weather_url = (
        "https://archive-api.open-meteo.com/v1/archive"
    )

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),

        "daily": ",".join([
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "wind_speed_10m_max",
            "precipitation_sum",
            "weathercode",
            "sunrise",
            "sunset"
        ]),

        "timezone": "auto"
    }

    response = requests.get(
        weather_url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    payload = response.json()

    daily = payload.get("daily", {})

    dates = daily.get("time", [])
    temperatures = daily.get(
        "temperature_2m_mean", []
    )
    humidity = daily.get(
        "relative_humidity_2m_mean", []
    )
    windspeed = daily.get(
        "wind_speed_10m_max", []
    )
    precipitation = daily.get(
        "precipitation_sum", []
    )
    weathercode = daily.get(
        "weathercode", []
    )
    sunrise = daily.get(
        "sunrise", []
    )
    sunset = daily.get(
        "sunset", []
    )

    # Check that the API returned data
    if not dates or not temperatures:
        raise ValueError(
            "No weather data was returned from Open-Meteo."
        )

    # Check that all weather columns have the same length
    lengths = [
        len(dates),
        len(temperatures),
        len(humidity),
        len(windspeed),
        len(precipitation),
        len(weathercode),
        len(sunrise),
        len(sunset)
    ]

    if len(set(lengths)) != 1:
        raise ValueError(
            "The weather data returned by the API "
            "has inconsistent lengths."
        )

    # Create DataFrame
    df = pd.DataFrame({
        "date": dates,
        "temperature": temperatures,
        "humidity": humidity,
        "windspeed": windspeed,
        "precipitation": precipitation,
        "weathercode": weathercode,
        "sunrise": sunrise,
        "sunset": sunset
    })

    # Format sunrise and sunset
    df["sunrise"] = pd.to_datetime(
        df["sunrise"]
    ).dt.strftime("%H:%M")

    df["sunset"] = pd.to_datetime(
        df["sunset"]
    ).dt.strftime("%H:%M")

    # Save data
    df.to_csv(
        output_csv,
        index=False
    )

    return df


# ==========================================
# LOAD WEATHER CSV
# ==========================================

def load_city_csv(
    city: str,
    folder: str = "."
) -> pd.DataFrame:
    """Load the historical weather CSV."""

    filename = "city_historical_weather.csv"

    filepath = f"{folder}/{filename}"

    try:
        df = pd.read_csv(filepath)

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Weather CSV for '{city}' was not found: "
            f"{filepath}"
        )

    return df


# ==========================================
# PREPARE MACHINE LEARNING FEATURES
# ==========================================

def prepare_features(
    df: pd.DataFrame,
    lags: int = 7
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """
    Create lagged temperature features.

    lag_1 = previous day's temperature
    lag_2 = temperature from 2 days ago
    etc.
    """

    if "temperature" not in df.columns:
        raise ValueError(
            "CSV must contain a 'temperature' column."
        )

    data = df[["temperature"]].copy()

    for i in range(1, lags + 1):
        data[f"lag_{i}"] = (
            data["temperature"].shift(i)
        )

    data = data.dropna()

    x = data[
        [f"lag_{i}" for i in range(1, lags + 1)]
    ].values

    y = data["temperature"].values

    return x, y, data.index


# ==========================================
# TRAIN MODEL
# ==========================================

def train_model(
    x: np.ndarray,
    y: np.ndarray
) -> LinearRegression:
    """Train a linear regression model."""

    model = LinearRegression()

    model.fit(x, y)

    return model


# ==========================================
# PREDICT NEXT DAYS
# ==========================================

def predict_next_days(
    model: LinearRegression,
    recent_temps: np.ndarray,
    days: int = 7
) -> list[float]:
    """
    Predict future temperatures using the
    most recent temperatures.
    """

    lags = len(recent_temps)

    predictions = []

    # Most recent temperature comes first
    window = list(recent_temps)

    for _ in range(days):

        x = np.array(
            window[:lags]
        ).reshape(1, -1)

        prediction = float(
            model.predict(x)[0]
        )

        predictions.append(prediction)

        # Add prediction as newest value
        window = (
            [prediction]
            + window[:lags - 1]
        )

    return predictions


# ==========================================
# PREDICT NEXT 7 DAYS FOR A CITY
# ==========================================

def predict_city_next_week(
    city: str,
    folder: str = "."
) -> pd.DataFrame:
    """Predict the next 7 days of temperature."""

    df = load_city_csv(
        city,
        folder
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    # Make sure data is chronological
    df = df.sort_values("date")

    x, y, _ = prepare_features(
        df,
        lags=7
    )

    model = train_model(
        x,
        y
    )

    temperatures = (
        df["temperature"]
        .dropna()
        .values
    )

    if len(temperatures) < 7:
        raise ValueError(
            f"Not enough temperature data for "
            f"{city}. Need at least 7 days."
        )

    # Most recent temperature first
    recent = temperatures[::-1][:7]

    predictions = predict_next_days(
        model,
        recent,
        days=7
    )

    last_date = df["date"].max()

    future_dates = [
        last_date + pd.Timedelta(days=i)
        for i in range(1, 8)
    ]

    result = pd.DataFrame({
        "date": future_dates,
        "predicted temperature": predictions
    })

    return result


# ==========================================
# RUN PROGRAM DIRECTLY
# ==========================================

if __name__ == "__main__":

    city = input(
        "Enter city name: "
    ).strip()

    if not city:
        print("Please enter a city name.")
        raise SystemExit

    try:

        # Download historical weather
        df = fetch_historical_weather(
            city,
            years=5,
            output_csv="city_historical_weather.csv"
        )

        # Predict next 7 days
        predictions = predict_city_next_week(
            city
        )

        print(
            f"\nPredicted temperatures "
            f"for the next 7 days in {city}:"
        )

        print(predictions)

        # Display most recent weather
        row = df.iloc[-1]

        print(
            f"\nMost recent weather data "
            f"for {city}:"
        )

        print(f"Date: {row['date']}")
        print(
            f"Temperature: "
            f"{row['temperature']}"
        )
        print(
            f"Humidity: "
            f"{row['humidity']}"
        )
        print(
            f"Windspeed: "
            f"{row['windspeed']}"
        )
        print(
            f"Precipitation: "
            f"{row['precipitation']}"
        )
        print(
            f"Weathercode: "
            f"{row['weathercode']}"
        )
        print(
            f"Sunrise: "
            f"{row['sunrise']}"
        )
        print(
            f"Sunset: "
            f"{row['sunset']}"
        )

    except Exception as e:

        print(
            f"Error processing weather "
            f"for '{city}': {e}"
        )



#WEATHER.PY