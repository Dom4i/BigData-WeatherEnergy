# %%
# Wetter (Temperatur) – City → AT wide pivot
# Ziel:
# - City-processed CSVs einlesen
# - nur Temperatur behalten
# - breit pivotieren (temp_c_<city>)
# - optionale Aggregate (mean/std)
# - stündlich, UTC, ISO-Z

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent  # Ordner von weather_processing_model2.py

CITIES_DIR = (BASE_DIR / "../../data/processed/Wetterdata/cities").resolve()
OUT_DIR    = (BASE_DIR / "../../data/processed/Wetterdata").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "weather_temp_hourly_AT_2015_2019_model2.csv"

CITIES = [
    "bregenz", "eisenstadt", "graz", "innsbruck", "klagenfurt",
    "linz", "salzburg", "st_poelten", "wien",
]

dfs = []
for city in CITIES:
    f = CITIES_DIR / f"{city}_hourly_2015_2019.csv"
    if not f.exists():
        raise FileNotFoundError(f"Fehlt: {f}")

    df = pd.read_csv(f)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df = df[["timestamp", "temp_c"]].copy()
    df = df.rename(columns={"temp_c": f"temp_c_{city}"})
    dfs.append(df)

df_wide = dfs[0]
for df in dfs[1:]:
    df_wide = df_wide.merge(df, on="timestamp", how="outer")

df_wide = df_wide.sort_values("timestamp").drop_duplicates("timestamp")

temp_cols = [c for c in df_wide.columns if c.startswith("temp_c_")]
df_wide["temp_c_mean"] = df_wide[temp_cols].mean(axis=1)
df_wide["temp_c_std"]  = df_wide[temp_cols].std(axis=1, ddof=0)

df_out = df_wide.copy()
df_out["timestamp"] = df_out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

df_out.to_csv(OUT_FILE, index=False)
print("Gespeichert:", OUT_FILE)
print("Zeilen:", len(df_out))

