# %%
# Feiertage / Kalender (AT) - Raw -> Processed
# Ziel:
# - Feiertags-CSV einlesen (date-Spalte)
# - hourly Kalender-Zeitreihe für 2015-2019 erzeugen
# - Features: weekday, is_weekend, is_holiday
# - als processed CSV speichern (Join auf timestamp)

from pathlib import Path
import pandas as pd

# %%
# KONFIGURATION
RAW_HOLIDAYS = Path("data/raw/Feiertage/feiertage_at_2015_2020_clean.csv")
OUT_DIR = Path("data/processed/Kalender")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_HOURLY = OUT_DIR / "calendar_hourly_AT_2015_2019.csv"

START_DATE = "2015-01-01"
END_DATE = "2019-12-31"

# %%
# FEIERTAGE EINLESEN
df_h = pd.read_csv(RAW_HOLIDAYS)

# date parsen
df_h["date"] = pd.to_datetime(df_h["date"], errors="coerce")
df_h = df_h.dropna(subset=["date"]).copy()

# nur Projektzeitraum (daily)
df_h = df_h[(df_h["date"] >= START_DATE) & (df_h["date"] <= END_DATE)].copy()

# Set mit Feiertags-Daten (für schnellen Lookup)
holiday_dates = set(df_h["date"].dt.date)

print("Feiertage im Zeitraum:", len(holiday_dates))

# %%
# HOURLY ZEITACHSE ERZEUGEN (UTC)
hours = pd.date_range(
    start=f"{START_DATE} 00:00:00",
    end=f"{END_DATE} 23:00:00",
    freq="H",
    tz="UTC"
)
df = pd.DataFrame({"timestamp": hours})

# %%
# FEATURES BERECHNEN
# weekday: 0=Mon, ..., 6=Sun
df["weekday"] = df["timestamp"].dt.weekday

# is_weekend: Samstag(5) oder Sonntag(6)
df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

# is_holiday: wenn das Datum in holiday_dates vorkommt
df["date"] = df["timestamp"].dt.date
df["is_holiday"] = df["date"].isin(holiday_dates).astype(int)



# Aufräumen
df = df.drop(columns=["date"])

# timestamp formatieren wie in euren anderen processed files
df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

# Spaltenreihenfolge
df = df[["timestamp", "weekday", "is_weekend", "is_holiday"]]

# %%
# SPEICHERN
df.to_csv(OUT_HOURLY, index=False)
print("Gespeichert:", OUT_HOURLY.resolve(), "| Zeilen:", len(df))
print(df.head(10).to_string(index=False))
