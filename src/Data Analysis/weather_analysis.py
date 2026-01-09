import pandas as pd
from pathlib import Path

# =========================================================
# Wetterdaten-Analyse Österreich
# =========================================================

base_path = Path("../../data/raw/Wetterdata")

all_data = []

for city_dir in base_path.iterdir():
    if city_dir.is_dir():
        city = city_dir.name  # Stadtname aus Ordner weil unterschiedlich geschrieben

        for file in city_dir.glob("*.csv"):
            try:
                # Dateiname
                filename_parts = file.stem.split("_")
                year = int(filename_parts[-2])
                month = int(filename_parts[-1])

                # CSV-Datei einlesen
                df = pd.read_csv(file)

                # Zusatzinformationen ergänzen
                df["city"] = city
                df["year"] = year
                df["month"] = month

                # Zeitstempel konvertieren
                df["time"] = pd.to_datetime(df["time"], errors="coerce")

                all_data.append(df)

            except Exception as e:
                print(f"Fehler beim Einlesen von {file.name}: {e}")

# ---------------------------------
# Alle Wetterdaten zusammenführen
# ---------------------------------
weather_df = pd.concat(all_data, ignore_index=True)

print("Vorschau der zusammengeführten Wetterdaten:")
print(weather_df.head(), "\n")

# ---------------------------------
# Vollständigkeit
# ---------------------------------
print("Fehlende Werte pro Spalte:")
print(weather_df.isnull().sum(), "\n")

print("Anzahl Datensätze insgesamt:", len(weather_df))
print("Anzahl Städte:", weather_df["city"].nunique(), "\n")

# ---------------------------------
# Zeitstempel
# ---------------------------------
print("Ungültige Zeitstempel:")
print(weather_df[weather_df["time"].isnull()], "\n")

# Doppelte Zeitstempel pro Stadt prüfen
duplicates = weather_df.duplicated(subset=["city", "time"])
print("Anzahl doppelter Zeitstempel:", duplicates.sum(), "\n")

# ---------------------------------
# Realistische Werte
# ---------------------------------
# Realistische Wertebereiche
invalid_temp = weather_df[(weather_df["temp"] < -30) | (weather_df["temp"] > 45)]
invalid_rhum = weather_df[(weather_df["rhum"] < 0) | (weather_df["rhum"] > 100)]
invalid_prcp = weather_df[weather_df["prcp"] < 0]
invalid_wind = weather_df[weather_df["wspd"] < 0]
invalid_pressure = weather_df[
    (weather_df["pres"].notnull()) &
    ((weather_df["pres"] < 850) | (weather_df["pres"] > 1100))
]

print("Unplausible Temperaturwerte:", len(invalid_temp))
print("Unplausible Luftfeuchtewerte:", len(invalid_rhum))
print("Negative Niederschlagswerte:", len(invalid_prcp))
print("Negative Windgeschwindigkeiten:", len(invalid_wind))
print("Unplausible Luftdruckwerte:", len(invalid_pressure), "\n")

# ---------------------------------
# Statistische Analyse
# ---------------------------------
print("Deskriptive Statistik (Temperatur, Feuchte, Wind, Luftdruck):")
print(
    weather_df[["temp", "rhum", "wspd", "pres"]].describe(),
    "\n"
)

# Durchschnittstemperatur pro Stadt
avg_temp_city = weather_df.groupby("city")["temp"].mean().sort_values()
print("Durchschnittstemperatur pro Stadt:")
print(avg_temp_city, "\n")

# ---------------------------------
# Korrelationen
# ---------------------------------
correlation_matrix = weather_df[
    ["temp", "rhum", "wspd", "pres"]
].corr()

print("Korrelationsmatrix:")
print(correlation_matrix, "\n")
