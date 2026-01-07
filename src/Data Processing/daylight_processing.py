# %%
# Tageslicht (AT) - Raw -> Processed
# Ziel:
# - date parsen
# - Tageslänge berechnen (daylight_seconds / daylight_hours)
# - als stündliche Zeitreihe speichern (für späteren Join)

from pathlib import Path
import pandas as pd

# %%
# KONFIGURATION
RAW_FILE = Path("data/raw/Tageslicht/austria_sunrise_sunset_avg_last5y_daily.csv")
OUT_DIR = Path("data/processed/Tageslicht")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DAILY = OUT_DIR / "daylight_daily_AT_2015_2019.csv"
OUT_HOURLY = OUT_DIR / "daylight_hourly_AT_2015_2019.csv"

START_DATE = "2015-01-01"
END_DATE = "2019-12-31"

# %%
# EINLESEN
df = pd.read_csv(RAW_FILE)

# %%
# DATE PARSEN + FILTERN
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).copy()

df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()
df = df.sort_values("date")

# %%
# FEATURE ENGINEERING: Tageslänge

df["daylight_seconds"] = df["sunset_avg_seconds"] - df["sunrise_avg_seconds"]
df["daylight_hours"] = df["daylight_seconds"] / 3600.0


# Nur sinnvolle Spalten behalten
df_daily = df[["date", "daylight_hours", "daylight_seconds"]].copy()

# Speichern (daily)
df_daily.to_csv(OUT_DAILY, index=False)
print("Gespeichert (daily):", OUT_DAILY.resolve(), "| Zeilen:", len(df_daily))

# %%
# Wir erstellen pro Tag 24 Stunden-Timestamps und hängen daylight_hours dran.
hours = pd.date_range(start=f"{START_DATE} 00:00:00", end=f"{END_DATE} 23:00:00", freq="H", tz="UTC")
df_hourly = pd.DataFrame({"timestamp": hours})

# Join: hourly timestamp -> date, dann mit df_daily matchen
df_hourly["date"] = df_hourly["timestamp"].dt.floor("D").dt.tz_localize(None)  # date ohne tz
df_hourly = df_hourly.merge(df_daily, on="date", how="left")

# timestamp formatieren wie bei euren anderen processed files
df_hourly["timestamp"] = df_hourly["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

# Aufräumen
df_hourly = df_hourly.drop(columns=["date"])
df_hourly = df_hourly[["timestamp", "daylight_hours", "daylight_seconds"]]

# Speichern (hourly)
df_hourly.to_csv(OUT_HOURLY, index=False)
print("Gespeichert (hourly):", OUT_HOURLY.resolve(), "| Zeilen:", len(df_hourly))
