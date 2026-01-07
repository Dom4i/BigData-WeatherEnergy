# %%
# Wetterdaten (Meteostat) - Raw -> Processed (pro Stadt)
# Ziel:
# 1) Alle Monats-CSV pro Stadt zusammenhängen (2015-01 bis 2019-12)
# 2) Nur relevante Features behalten: time/temp/rhum/wspd/wpgt/tsun
# 3) Zeit in gleiches Format wie Strom bringen: timestamp als ISO-UTC-String mit Z
# 4) Pro Stadt ein processed CSV abspeichern

from pathlib import Path
import pandas as pd

# %%
# KONFIGURATION
RAW_WEATHER_DIR = Path("data/raw/Wetterdata")
OUT_DIR = Path("data/processed/Wetterdata/cities")
OUT_DIR.mkdir(parents=True, exist_ok=True)


CITIES = [
    "bregenz",
    "eisenstadt",
    "graz",
    "innsbruck",
    "klagenfurt",
    "Linz",
    "Salzburg",
    "st_poelten",
    "wien",
]

# Projekt-Zeitraum (UTC)
START = "2015-01-01 00:00:00"
END = "2019-12-31 23:00:00"

# Zu behaltende Spalten aus den Rohdaten
KEEP_COLS_RAW = ["time", "temp", "prcp", "rhum", "wspd", "pres"]

# Output-Dateiname pro Stadt
def out_file_for_city(city: str) -> Path:
    return OUT_DIR / f"{city}_hourly_2015_2019.csv"

# %%
# Funktion: alle Monatsdateien für eine Stadt finden
def find_monthly_files(city: str) -> list[Path]:
    city_dir = RAW_WEATHER_DIR / city
    if not city_dir.exists():
        raise FileNotFoundError(f"Ordner nicht gefunden: {city_dir}")


    files = sorted(city_dir.glob(f"{city}_*.csv"))
    if not files:
        raise FileNotFoundError(f"Keine CSV-Dateien gefunden für {city} in {city_dir}")

    return files

# %%
# Hauptverarbeitung
start_ts = pd.Timestamp(START, tz="UTC")
end_ts = pd.Timestamp(END, tz="UTC")

for city in CITIES:
    print(f"\n--- Verarbeite Stadt: {city} ---")

    files = find_monthly_files(city)
    print(f"Gefundene Monatsfiles: {len(files)}")

    # Alle Monatsfiles einlesen & anhängen
    dfs = []
    for f in files:
        df_m = pd.read_csv(f)

        # Prüfen, ob alle benötigten Spalten existieren
        missing = [c for c in KEEP_COLS_RAW if c not in df_m.columns]
        if missing:
            raise ValueError(f"Datei {f.name} fehlt Spalten: {missing}")

        # Nur benötigte Spalten nehmen
        df_m = df_m[KEEP_COLS_RAW].copy()
        dfs.append(df_m)

    df_city = pd.concat(dfs, ignore_index=True)

    # %%
    # Zeit parsen
    if "timestamp" not in df_city.columns:
        if "time" not in df_city.columns:
            raise KeyError(f"Erwarte 'time' oder 'timestamp'. Vorhandene Spalten: {list(df_city.columns)}")
        df_city["timestamp"] = pd.to_datetime(df_city["time"], errors="coerce", utc=True)

    # Ungültige Zeiten entfernen
    df_city = df_city.dropna(subset=["timestamp"]).copy()

    # Auf Projekt-Zeitraum filtern
    df_city = df_city[(df_city["timestamp"] >= start_ts) & (df_city["timestamp"] <= end_ts)].copy()

    # Doppelte Stunden entfernen
    df_city = df_city.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first")

    # %%
    # Spalten final umbenennen / aufräumen
    if "time" in df_city.columns:
        df_city = df_city.drop(columns=["time"])

    # Abkürzungen in verständlichere Namen umbenennen
    rename_map = {
        "temp": "temp_c",
        "prcp": "precip_mm",
        "rhum": "humidity_pct",
        "wspd": "wind_speed_kmh",
        "pres": "pressure_hpa",
    }
    df_city = df_city.rename(columns=rename_map)
    # Stadtspalte mitgeben (für spätere Debugs / Merge)
    df_city["city"] = city

    # Spaltenreihenfolge
    df_city = df_city[["timestamp", "city", "temp_c", "precip_mm", "humidity_pct", "wind_speed_kmh", "pressure_hpa"]]

    # %%
    # Timestamp in gleiches Format wie Strom bringen (ISO UTC mit Z)
    df_out = df_city.copy()
    df_out["timestamp"] = df_out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Speichern
    out_path = out_file_for_city(city)
    df_out.to_csv(out_path, index=False)

    print(f"Gespeichert: {out_path} | Zeilen: {len(df_out)}")
    print("Von/Bis:", df_city["timestamp"].min(), "->", df_city["timestamp"].max())

print("\nFertig.")

# %%
# AT-AGGREGATION: Mittelwert über alle Städte pro Stunde
# Ergebnis: eine Zeile pro Stunde für ganz Österreich


# Alle processed City-Files einlesen
city_files = [out_file_for_city(city) for city in CITIES]

dfs = []
for f in city_files:
    if not f.exists():
        raise FileNotFoundError(f"Processed City-File fehlt: {f}")
    df_c = pd.read_csv(f)

    # timestamp parsen (ist aktuell ISO-String mit Z)
    df_c["timestamp"] = pd.to_datetime(df_c["timestamp"], utc=True, errors="coerce")
    df_c = df_c.dropna(subset=["timestamp"]).copy()

    dfs.append(df_c)

df_all = pd.concat(dfs, ignore_index=True)

# Mittelwert pro Stunde über alle Städte berechnen
df_at = (
    df_all
    .groupby("timestamp", as_index=False)[
        ["temp_c", "precip_mm", "humidity_pct", "wind_speed_kmh", "pressure_hpa"]
    ]
    .mean()
)


# Spaltenreihenfolge
df_at = df_at[["timestamp", "temp_c", "precip_mm", "humidity_pct", "wind_speed_kmh", "pressure_hpa"]]

# Timestamp wieder ins gleiche Format wie Strom bringen (ISO UTC mit Z)
df_at["timestamp"] = df_at["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

# Speichern
AT_OUT_FILE = OUT_DIR.parent / "weather_hourly_AT_2015_2019.csv"  # data/processed/Wetterdata/weather_hourly_AT_2015_2019.csv
df_at.to_csv(AT_OUT_FILE, index=False)

print("AT-File gespeichert:", AT_OUT_FILE.resolve(), "| Zeilen:", len(df_at))


