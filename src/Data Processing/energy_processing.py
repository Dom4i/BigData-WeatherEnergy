# %%
# Stromverbrauch Österreich (OPSD) - Raw -> Processed
# Ziel:
# - große OPSD-CSV einlesen
# - nur Österreich-Spalten behalten (AT)
# - Zeitstempel (utc_timestamp) sauber als UTC parsen
# - auf Projekt-Zeitraum 2015-01-01 bis 2019-12-31 filtern
# - Spalten umbenennen (einheitliches Schema)
# - als processed CSV abspeichern (InfluxDB-ready)

from pathlib import Path
import pandas as pd

# %%
# -----------------------------
# KONFIGURATION
# -----------------------------
RAW_ENERGY_CSV = Path("data/raw/Strom/time_series_60min_singleindex.csv")
OUT_DIR = Path("data/processed/Strom")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(RAW_ENERGY_CSV.exists())
OUT_FILE = OUT_DIR / "energy_hourly_AT_2015_2019_processed.csv"

# Projekt-Zeitraum (UTC)
START = "2015-01-01 00:00:00"
END = "2019-12-31 23:00:00"

# %%
# CSV EINLESEN
df = pd.read_csv(RAW_ENERGY_CSV)

# %%
# NUR DIE FEATURES BEHALTEN, DIE WIR BRAUCHEN
keep_cols = [
    "utc_timestamp",
    "AT_load_actual_entsoe_transparency",
    "AT_solar_generation_actual",
    "AT_wind_onshore_generation_actual",
    # optional, falls ihr es später doch wollt:
    # "AT_price_day_ahead",
    # "AT_load_forecast_entsoe_transparency",
]

# Prüfen, ob alle Spalten existieren (sonst Tippfehler / falsche Datei)
missing = [c for c in keep_cols if c not in df.columns]
if missing:
    raise ValueError(f"Diese erwarteten Spalten fehlen in der CSV: {missing}")

df = df[keep_cols].copy()


# %%
# TIMESTAMP ALS UTC PARSEN
# Beispiel-Format: 2015-01-01T00:00:00Z
df["timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True, errors="coerce")

# Ungültige Zeitstempel entfernen
df = df.dropna(subset=["timestamp"]).copy()

# OPTIONAL: Originalspalte entfernen (nur wenn du sie nicht mehr brauchst)
df = df.drop(columns=["utc_timestamp"])

# %%
# AUF PROJEKT-ZEITRAUM FILTERN
start_ts = pd.Timestamp(START, tz="UTC")
end_ts = pd.Timestamp(END, tz="UTC")

df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)].copy()

# Sortieren und doppelte Zeitstempel entfernen
df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first")

# %%
# SPALTEN EINHEITLICH BENENNEN
rename_map = {
    "utc_timestamp": "timestamp",
    "AT_load_actual_entsoe_transparency": "load_mw",
    "AT_solar_generation_actual": "solar_mw",
    "AT_wind_onshore_generation_actual": "wind_mw",
    # optional:
    # "AT_price_day_ahead": "price_day_ahead",
    # "AT_load_forecast_entsoe_transparency": "load_forecast_mw",
}
df = df.rename(columns=rename_map)


# Spalten-Reihenfolge festlegen
ordered_cols = ["timestamp", "load_mw"]
for c in ["solar_mw", "wind_mw", "price_day_ahead", "load_forecast_mw"]:
    if c in df.columns:
        ordered_cols.append(c)

df = df[ordered_cols]

# %%
# ALS PROCESSED CSV SPEICHERN
# Für DB-Import ist ein ISO-UTC-String oft am einfachsten (mit Z am Ende)
df_out = df.copy()
df_out["timestamp"] = df_out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

df_out.to_csv(OUT_FILE, index=False)
print("Gespeichert:", OUT_FILE.resolve())
