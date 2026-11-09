from flask import Flask, render_template, request, jsonify 
from flask_cors import CORS
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import date, timedelta
import os

app = Flask(__name__)
CORS(app)

# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# CROP INFORMATION
# =========================================================

@app.route("/crop-info", methods=["POST"])
def crop_info():
    data = request.get_json(silent=True) or {}
    crop = str(data.get("crop", "")).strip()

    if not crop:
        return jsonify({
            "success": False,
            "error": "Please enter a crop name."
        }), 400

    try:
        url = "https://en.wikipedia.org/w/api.php"
        headers = {"User-Agent": "AgricultureDashboard/1.0"}

        search_params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": crop,
            "srlimit": 5
        }

        response = requests.get(
            url,
            params=search_params,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        search_data = response.json()

        results = search_data.get("query", {}).get("search", [])

        if not results:
            return jsonify({
                "success": False,
                "error": f"No Wikipedia page found for '{crop}'."
            }), 404

        page_title = results[0]["title"]

        summary_params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "exsentences": 5,
            "titles": page_title
        }

        summary_response = requests.get(
            url,
            params=summary_params,
            headers=headers,
            timeout=10
        )
        summary_response.raise_for_status()

        summary_data = summary_response.json()
        pages = summary_data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        summary = page.get("extract", "No summary available.")

        return jsonify({
            "success": True,
            "crop": page_title,
            "summary": summary
        })

    except requests.RequestException:
        app.logger.exception("Wikipedia request failed")
        return jsonify({
            "success": False,
            "error": "Could not connect to Wikipedia. Please try again later."
        }), 502

    except (ValueError, KeyError, StopIteration):
        app.logger.exception("Could not process Wikipedia response")
        return jsonify({
            "success": False,
            "error": "Could not process crop information."
        }), 502


# =========================================================
# PRICE ANALYSIS
# =========================================================

@app.route("/price-analysis", methods=["POST"])
def price_analysis():
    data = request.get_json(silent=True) or {}

    crop = str(data.get("crop", "")).strip()

    # Location is optional. If omitted, analyze all locations.
    location = str(data.get("location") or "").strip()

    if not crop:
        return jsonify({
            "success": False,
            "error": "Please enter a crop name."
        }), 400

    try:
        program_folder = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(program_folder, "crop_price.csv")

        if not os.path.exists(csv_path):
            return jsonify({
                "success": False,
                "error": "crop_price.csv could not be found."
            }), 500

        df = pd.read_csv(csv_path)

        required_columns = ["Crop", "Location", "Date", "Price"]
        missing_columns = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            return jsonify({
                "success": False,
                "error": (
                    "Missing column(s) in crop_price.csv: "
                    + ", ".join(missing_columns)
                )
            }), 500

        # Match crop names without case sensitivity.
        crop_mask = (
            df["Crop"].astype(str).str.strip().str.lower()
            == crop.lower()
        )

        crop_df = df[crop_mask].copy()

        # Filter by location only if a location was supplied.
        if location:
            location_mask = (
                crop_df["Location"].astype(str).str.strip().str.lower()
                == location.lower()
            )
            crop_df = crop_df[location_mask].copy()

        if crop_df.empty:
            scope = (
                f" in {location}"
                if location
                else " across available locations"
            )

            return jsonify({
                "success": False,
                "error": (
                    f"No price data found for {crop}{scope}. "
                    "Check the Crop and Location values in crop_price.csv."
                )
            }), 404

        crop_df["Date"] = pd.to_datetime(
            crop_df["Date"], errors="coerce"
        )

        crop_df["Price"] = pd.to_numeric(
            crop_df["Price"], errors="coerce"
        )

        crop_df = crop_df.dropna(subset=["Date", "Price"])

        if crop_df.empty:
            return jsonify({
                "success": False,
                "error": "The Date or Price values are invalid."
            }), 400

        crop_df = crop_df.sort_values("Date")

        avg_price = float(crop_df["Price"].mean())
        max_price = float(crop_df["Price"].max())
        min_price = float(crop_df["Price"].min())
        latest_price = float(crop_df.iloc[-1]["Price"])

        graph_dates = [
            value.strftime("%Y-%m-%d")
            for value in crop_df["Date"]
        ]

        graph_prices = [
            float(value)
            for value in crop_df["Price"]
        ]

        return jsonify({
            "success": True,
            "crop": crop,
            "location": location,
            "average": round(avg_price, 2),
            "maximum": round(max_price, 2),
            "minimum": round(min_price, 2),
            "latest": round(latest_price, 2),
            "records": len(crop_df),
            "dates": graph_dates,
            "prices": graph_prices
        })

    except Exception:
        app.logger.exception("Price analysis failed")
        return jsonify({
            "success": False,
            "error": "Could not analyze crop prices. Check the server log."
        }), 500


# =========================================================
# WEATHER FUNCTIONS
# =========================================================

def get_city_coordinates(city):
    url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()

    results = response.json().get("results")

    if not results:
        raise ValueError(f"Could not find city: {city}")

    location = results[0]

    return (
        float(location["latitude"]),
        float(location["longitude"])
    )


def fetch_historical_weather(city):
    lat, lon = get_city_coordinates(city)

    end_date = date.today()
    start_date = end_date - timedelta(days=int(5 * 365.25))

    weather_url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": ",".join([
            "temperature_2m_mean",
            "wind_speed_10m_max",
            "precipitation_sum",
            "weather_code",
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

    if response.status_code != 200:
        raise ValueError("The weather service returned an error.")

    payload = response.json()
    daily = payload.get("daily", {})
    dates = daily.get("time", [])

    df = pd.DataFrame({
        "date": dates,
        "temperature": daily.get("temperature_2m_mean", []),
        "humidity": [None] * len(dates),
        "windspeed": daily.get("wind_speed_10m_max", []),
        "precipitation": daily.get("precipitation_sum", []),
        "weathercode": daily.get("weather_code", []),
        "sunrise": daily.get("sunrise", []),
        "sunset": daily.get("sunset", [])
    })

    return df


# =========================================================
# WEATHER PREDICTION
# =========================================================

@app.route("/weather", methods=["POST"])
def weather():
    data = request.get_json(silent=True) or {}
    city = str(data.get("city", "")).strip()

    if not city:
        return jsonify({
            "success": False,
            "message": "Please enter a city."
        }), 400

    try:
        df = fetch_historical_weather(city)

        if df.empty:
            return jsonify({
                "success": False,
                "message": "No weather data was returned."
            }), 502

        df["date"] = pd.to_datetime(
            df["date"], errors="coerce"
        )

        df = df.dropna(
            subset=["date", "temperature"]
        ).copy()

        df = df.sort_values("date")

        if len(df) < 10:
            return jsonify({
                "success": False,
                "message": "Not enough weather data to make a prediction."
            }), 400

        temperatures = df["temperature"].astype(float).values

        X = []
        y = []

        for i in range(7, len(temperatures)):
            X.append(temperatures[i - 7:i][::-1])
            y.append(temperatures[i])

        X = np.array(X)
        y = np.array(y)

        model = LinearRegression()
        model.fit(X, y)

        window = list(temperatures[-7:][::-1])
        predictions = []

        for _ in range(7):
            x = np.array(
                window, dtype=float
            ).reshape(1, -1)

            prediction = float(model.predict(x)[0])
            predictions.append(round(prediction, 2))
            window = [prediction] + window[:-1]

        last_date = df["date"].iloc[-1]

        future_dates = [
            (
                last_date + pd.Timedelta(days=i)
            ).strftime("%Y-%m-%d")
            for i in range(1, 8)
        ]

        latest = df.iloc[-1]

        def format_time(value):
            if pd.isna(value):
                return None

            value = str(value)

            if "T" in value:
                return value.split("T", 1)[1][:5]

            return value

        latest_temperature = (
            float(latest["temperature"])
            if pd.notna(latest["temperature"])
            else None
        )

        latest_windspeed = (
            float(latest["windspeed"])
            if pd.notna(latest["windspeed"])
            else None
        )

        latest_precipitation = (
            float(latest["precipitation"])
            if pd.notna(latest["precipitation"])
            else None
        )

        return jsonify({
            "success": True,
            "city": city,
            "latest": {
                "date": latest["date"].strftime("%Y-%m-%d"),
                "temperature": latest_temperature,
                "humidity": None,
                "windspeed": latest_windspeed,
                "precipitation": latest_precipitation,
                "sunrise": format_time(latest["sunrise"]),
                "sunset": format_time(latest["sunset"])
            },
            "dates": future_dates,
            "predictions": predictions,
            "predicted_temperatures": predictions
        })

    except requests.RequestException:
        app.logger.exception("Weather service request failed")

        return jsonify({
            "success": False,
            "message": (
                "Could not connect to the weather service. "
                "Please try again."
            )
        }), 502

    except Exception as e:
        app.logger.exception("Weather prediction failed")

        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


# =========================================================
# START FLASK
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)





#APP.PY