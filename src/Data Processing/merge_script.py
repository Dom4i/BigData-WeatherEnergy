# %%
# Merge-Script: Energy + Weather + Daylight + Calendar  -> features_hourly_AT_2015_2019
# Ziel:
# - 4 processed CSVs einlesen
# - timestamp überall als UTC datetime parsen
# - auf timestamp mergen (stündlich)
# - sauberes "features" CSV (und optional Parquet) schreiben

from pathlib import Path
import pandas as pd

# %%
# KONFIGURATION
ENERGY_FILE   = Path("data/processed/Strom/energy_hourly_AT_2015_2019_processed.csv")
WEATHER_FILE  = Path("data/processed/Wetterdata/weather_hourly_AT_2015_2019.csv")
DAYLIGHT_FILE = Path("data/processed/Tageslicht/daylight_hourly_AT_2015_2019.csv")
CAL_FILE      = Path("data/processed/Kalender/calendar_hourly_AT_2015_2019.csv")

OUT_DIR = Path("data/processed/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV     = OUT_DIR / "features_hourly_AT_2015_2019.csv"
OUT_PARQUET = OUT_DIR / "features_hourly_AT_2015_2019.parquet"  # für spark/ml

# %%
# CSV laden + timestamp parsen
def load_with_timestamp(path: Path, ts_col: str = "timestamp") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")

    df = pd.read_csv(path)

    if ts_col not in df.columns:
        raise KeyError(f"Spalte '{ts_col}' fehlt in {path.name}. Spalten: {list(df.columns)}")

    # timestamp robust parsen (ISO mit Z / ohne Z / etc.)
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col]).copy()

    # doppelte timestamps entfernen (sollte nicht vorkommen, aber sicher ist sicher)
    df = df.sort_values(ts_col).drop_duplicates(subset=[ts_col], keep="first")

    return df

# %%
# DATEN LADEN
df_energy   = load_with_timestamp(ENERGY_FILE, "timestamp")
df_weather  = load_with_timestamp(WEATHER_FILE, "timestamp")
df_daylight = load_with_timestamp(DAYLIGHT_FILE, "timestamp")
df_cal      = load_with_timestamp(CAL_FILE, "timestamp")

print("Energy:", df_energy.shape, "|", df_energy.columns.tolist())
print("Weather:", df_weather.shape, "|", df_weather.columns.tolist())
print("Daylight:", df_daylight.shape, "|", df_daylight.columns.tolist())
print("Calendar:", df_cal.shape, "|", df_cal.columns.tolist())


# %%
# -----------------------------
# MERGE (LEFT JOIN auf Energy als Basis)
# -----------------------------
df = df_energy.merge(df_weather, on="timestamp", how="left")
df = df.merge(df_daylight, on="timestamp", how="left")
df = df.merge(df_cal, on="timestamp", how="left")


# %%
# -----------------------------
# QUICK CHECKS
# -----------------------------
print("\nMerged shape:", df.shape)
print("Zeitraum:", df["timestamp"].min(), "->", df["timestamp"].max())

print("\nFehlende Werte pro Spalte (Top 15):")
na_counts = df.isna().sum().sort_values(ascending=False)
print(na_counts.head(15).to_string())

print("\nBeispiel (5 Zeilen):")
print(df.head(5).to_string(index=False))

# %%
# timestamp formatieren + speichern
df_out = df.copy()
df_out["timestamp"] = df_out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

df_out.to_csv(OUT_CSV, index=False)
print("\nGespeichert CSV:", OUT_CSV.resolve(), "| Zeilen:", len(df_out))

# Parquet für Spark/ML
try:
    df.to_parquet(OUT_PARQUET, index=False)
    print("Gespeichert Parquet:", OUT_PARQUET.resolve())
except Exception as e:
    print("Parquet konnte nicht gespeichert werden:", e)
