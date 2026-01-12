# %%
# Tageslicht (AT) - Raw -> Processed
# Ziel:
# - date parsen
# - sunrise/sunset seconds + Tageslänge berechnen
# - stündliche Zeitreihe erzeugen (UTC-Timestamps)
# - DST-korrekte Features in Lokalzeit (Europe/Vienna) berechnen:
#     hours_since_sunrise (negativ erlaubt)
#     hours_until_sunset
#     is_daylight

from pathlib import Path
import pandas as pd

# %%
# BASISPFAD: immer relativ zum Script (nicht zum Working Directory)
BASE_DIR = Path(__file__).resolve().parent

# %%
# KONFIGURATION (robust aufgelöst)
RAW_FILE = (BASE_DIR / "../../data/raw/Tageslicht/austria_sunrise_sunset_avg_last5y_daily.csv").resolve()

OUT_DIR = (BASE_DIR / "../../data/processed/Tageslicht").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DAILY  = OUT_DIR / "daylight_daily_AT_2015_2019_model2.csv"
OUT_HOURLY = OUT_DIR / "daylight_hourly_AT_2015_2019_model2.csv"

START_DATE = "2015-01-01"
END_DATE   = "2019-12-31"

LOCAL_TZ = "Europe/Vienna"

print("RAW_FILE:", RAW_FILE, "| exists:", RAW_FILE.exists())
print("OUT_DIR :", OUT_DIR)

# %%
# EINLESEN
df = pd.read_csv(RAW_FILE)

# %%
# DATE PARSEN + FILTERN
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).copy()

end_daily = pd.to_datetime(END_DATE) + pd.Timedelta(days=1)
df = df[(df["date"] >= START_DATE) & (df["date"] <= end_daily)].copy()
df = df.sort_values("date")

# %%
# REQUIRED COLS CHECK
required = {"sunrise_avg_seconds", "sunset_avg_seconds"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Im Daylight-CSV fehlen Spalten: {sorted(missing)}")

# %%
# FEATURE ENGINEERING (daily)
df["daylight_seconds"] = df["sunset_avg_seconds"] - df["sunrise_avg_seconds"]
df["daylight_hours"] = df["daylight_seconds"] / 3600.0

# Daily-Spalten (WICHTIG: sunrise/sunset seconds behalten!)
df_daily = df[
    ["date", "sunrise_avg_seconds", "sunset_avg_seconds", "daylight_hours", "daylight_seconds"]
].copy()

# Speichern (daily)
df_daily.to_csv(OUT_DAILY, index=False)
print("Gespeichert (daily):", OUT_DAILY.resolve(), "| Zeilen:", len(df_daily))

# %%
# HOURLY: UTC Zeitachse erzeugen (für Join mit Energy/Weather)
hours_utc = pd.date_range(
    start=f"{START_DATE} 00:00:00",
    end=f"{END_DATE} 23:00:00",
    freq="h",
    tz="UTC",
)
df_hourly = pd.DataFrame({"timestamp": hours_utc})

# %%
# Lokalzeit ableiten (DST-korrekt!)
ts_local = df_hourly["timestamp"].dt.tz_convert(LOCAL_TZ)

df_hourly["local_date"] = ts_local.dt.floor("D").dt.tz_localize(None)  # naive local date
df_hourly["local_seconds"] = (
    ts_local.dt.hour * 3600 + ts_local.dt.minute * 60 + ts_local.dt.second
).astype(int)

# %%
# Merge: stündliche Zeilen bekommen sunrise/sunset pro lokalem Tag
df_hourly = df_hourly.merge(
    df_daily,
    left_on="local_date",
    right_on="date",
    how="left",
)

# Nach dem Merge können am Rand (wegen UTC->lokal) 1-2 Stunden NaNs entstehen.
# Wir füllen diese mit dem letzten verfügbaren Tageswert (ffill) bzw. falls nötig am Anfang (bfill).
fill_cols = ["sunrise_avg_seconds", "sunset_avg_seconds", "daylight_hours", "daylight_seconds"]

df_hourly = df_hourly.sort_values("timestamp")
df_hourly[fill_cols] = df_hourly[fill_cols].ffill().bfill()

# Safety check
# Safety check (nach Fill)
if df_hourly[["sunrise_avg_seconds", "sunset_avg_seconds"]].isna().any().any():
    na_rows = df_hourly[df_hourly["sunrise_avg_seconds"].isna() | df_hourly["sunset_avg_seconds"].isna()]
    raise ValueError(
        "Fehlende Sunrise/Sunset-Werte auch nach Fill. Beispiel-Zeilen:\n"
        + na_rows.head(5).to_string(index=False)
    )


# %%
# RELATIVE FEATURES (negativ erlaubt)
df_hourly["hours_since_sunrise"] = (
    (df_hourly["local_seconds"] - df_hourly["sunrise_avg_seconds"]) / 3600.0
)

df_hourly["hours_until_sunset"] = (
    (df_hourly["sunset_avg_seconds"] - df_hourly["local_seconds"]) / 3600.0
)

df_hourly["is_daylight"] = (
    (df_hourly["local_seconds"] >= df_hourly["sunrise_avg_seconds"])
    & (df_hourly["local_seconds"] < df_hourly["sunset_avg_seconds"])
).astype(int)

# %%
# Output-Format (ISO UTC mit Z)
df_out = df_hourly.copy()
df_out["timestamp"] = df_out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

df_out = df_out[
    [
        "timestamp",
        "daylight_hours",
        "daylight_seconds",
        "hours_since_sunrise",
        "hours_until_sunset",
        "is_daylight",
    ]
]

df_out.to_csv(OUT_HOURLY, index=False)
print("Gespeichert (hourly):", OUT_HOURLY.resolve(), "| Zeilen:", len(df_out))
print(df_out.head(5).to_string(index=False))
