# ==========================================
# CROP INFORMATION + PRICE ANALYSIS
# ==========================================

import os
import requests
import pandas as pd


# ==========================================
# GET CROP INFORMATION FROM WIKIPEDIA
# ==========================================

def get_crop_info(crop: str) -> dict:
    """
    Search Wikipedia for a crop and return
    the page title and summary.
    """

    crop = crop.strip()

    if not crop:
        raise ValueError(
            "Please enter a crop name."
        )

    url = "https://en.wikipedia.org/w/api.php"

    headers = {
        "User-Agent": (
            "CropPriceTrendAnalyzer/1.0 "
            "(itsshlokareddy@gmail.com)"
        )
    }

    # Search Wikipedia
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

    data = response.json()

    search_results = (
        data
        .get("query", {})
        .get("search", [])
    )

    if not search_results:
        raise ValueError(
            f"No Wikipedia page found for '{crop}'."
        )

    # Best matching page
    page_title = search_results[0]["title"]

    # Get summary
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

    pages = (
        summary_data
        .get("query", {})
        .get("pages", {})
    )

    page = next(iter(pages.values()))

    summary = page.get(
        "extract",
        "No summary available."
    )

    return {
        "crop": crop,
        "page_title": page_title,
        "summary": summary
    }


# ==========================================
# LOAD CROP PRICE DATA
# ==========================================

def load_crop_data(
    csv_filename: str = "crop_price.csv"
) -> pd.DataFrame:
    """Load crop price data from the CSV file."""

    program_folder = os.path.dirname(
        os.path.abspath(__file__)
    )

    csv_path = os.path.join(
        program_folder,
        csv_filename
    )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find {csv_filename}.\n"
            f"Expected location:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = [
        "Crop",
        "Location",
        "Date",
        "Price"
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' is missing "
                f"from {csv_filename}."
            )

    return df


# ==========================================
# ANALYZE CROP PRICE
# ==========================================

def analyze_price(
    crop: str,
    location: str
) -> dict:
    """
    Analyze crop prices for a specific
    crop and location.
    """

    crop = crop.strip()
    location = location.strip()

    if not crop:
        raise ValueError(
            "Please enter a crop name."
        )

    if not location:
        raise ValueError(
            "Please enter a location."
        )

    # Load CSV
    df = load_crop_data()

    # Filter crop and location
    crop_df = df[
        (
            df["Crop"]
            .astype(str)
            .str.strip()
            .str.lower()
            == crop.lower()
        )
        &
        (
            df["Location"]
            .astype(str)
            .str.strip()
            .str.lower()
            == location.lower()
        )
    ].copy()

    if crop_df.empty:
        raise ValueError(
            f"No price data found for "
            f"{crop} in {location}."
        )

    # Convert dates
    crop_df["Date"] = pd.to_datetime(
        crop_df["Date"],
        errors="coerce"
    )

    # Convert prices
    crop_df["Price"] = pd.to_numeric(
        crop_df["Price"],
        errors="coerce"
    )

    # Remove invalid rows
    crop_df = crop_df.dropna(
        subset=["Date", "Price"]
    )

    if crop_df.empty:
        raise ValueError(
            "The Date or Price values "
            "in the CSV are invalid."
        )

    # Sort by date
    crop_df = crop_df.sort_values(
        "Date"
    )

    # Calculate statistics
    average_price = crop_df["Price"].mean()
    maximum_price = crop_df["Price"].max()
    minimum_price = crop_df["Price"].min()
    latest_price = crop_df.iloc[-1]["Price"]

    return {
        "crop": crop,
        "location": location,
        "average_price": float(average_price),
        "maximum_price": float(maximum_price),
        "minimum_price": float(minimum_price),
        "latest_price": float(latest_price),
        "number_of_records": len(crop_df),
        "dates": crop_df["Date"].dt.strftime(
            "%Y-%m-%d"
        ).tolist(),
        "prices": crop_df["Price"].tolist()
    }



#MARKET TRENDS