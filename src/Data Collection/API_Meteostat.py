import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()


# CONFIG

#cities wird nicht verwendet, oder?
cities = {
    "wien": (48.2082, 16.3738),
    "st_poelten": (48.2049, 15.6256),
    "linz": (48.3069, 14.2858),
    "salzburg": (47.8095, 13.0550),
    "graz": (47.0707, 15.4395),
    "klagenfurt": (46.6247, 14.3053),
    "innsbruck": (47.2692, 11.4041),
    "bregenz": (47.5031, 9.7471),
    "eisenstadt": (47.8456, 16.5233),
}

# Die Landeshauptstädte wurden manuell eingegeben, da mit einem API key das Anfragelimit für ganz Österreich überschritten wird
# Daher statt Loop über alle Bundesländer jedes einzeln gemacht und dazwischen API key geändert
CITY_NAME = "eisenstadt"
LAT = 47.8456
LON = 16.5233

START_DATE = "2015-01-01"
END_DATE = "2019-12-31"

API_KEY = os.getenv("RAPIDAPI_KEY_marcel")

BASE_URL = "https://meteostat.p.rapidapi.com/point/hourly"

HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": "meteostat.p.rapidapi.com"
}

OUTPUT_DIR = Path("../../data/raw/Wetterdata") / CITY_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# HELPER FUNCTIONS
def daterange(start, end):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)

def month_start_dates(start, end):
    dates = []
    current = start.replace(day=1)
    while current <= end:
        dates.append(current)
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return dates


# MAIN LOGIC
start_dt = datetime.fromisoformat(START_DATE)
end_dt = datetime.fromisoformat(END_DATE)

for month_start in month_start_dates(start_dt, end_dt):
    # Monatsende bestimmen
    if month_start.month == 12:
        next_month = month_start.replace(year=month_start.year + 1, month=1)
    else:
        next_month = month_start.replace(month=month_start.month + 1)

    month_end = min(next_month - timedelta(days=1), end_dt)

    print(f" Lade {CITY_NAME} {month_start.strftime('%Y-%m')}")

    all_chunks = []
    chunk_start = month_start

    while chunk_start <= month_end:
        chunk_end = min(chunk_start + timedelta(days=29), month_end)

        params = {
            "lat": LAT,
            "lon": LON,
            "start": chunk_start.strftime("%Y-%m-%d"),
            "end": chunk_end.strftime("%Y-%m-%d")
        }

        response = requests.get(BASE_URL, headers=HEADERS, params=params)
        response.raise_for_status()

        data = response.json()["data"]
        if data:
            df_chunk = pd.DataFrame(data)
            all_chunks.append(df_chunk)

        chunk_start = chunk_end + timedelta(days=1)

    if all_chunks:
        df_month = pd.concat(all_chunks, ignore_index=True)
        df_month["time"] = pd.to_datetime(df_month["time"])

        filename = f"{CITY_NAME}_{month_start.year}_{month_start.month:02d}.csv"
        df_month.to_csv(OUTPUT_DIR / filename, index=False)

        print(f"Gespeichert: {filename} ({len(df_month)} Zeilen)")
    else:
        print(f"Keine Daten für {month_start.strftime('%Y-%m')}")

print("Fertig!")
