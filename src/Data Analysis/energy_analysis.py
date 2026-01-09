import pandas as pd

# =========================================================
# Analyse Stromdaten in AT
# =========================================================

# Pfad zur Datei
file_path = "../../data/raw/Strom/time_series_60min_singleindex.csv"

# CSV-Datei einlesen
df = pd.read_csv(file_path)

print("Vorschau der Rohdaten:")
print(df.head(), "\n")

# ---------------------------------
# Auswahl der relevanten Spalten (AT)
# ---------------------------------
# Nur österreichische Werte verwendet
at_columns = [
    "utc_timestamp",
    "cet_cest_timestamp",
    "AT_load_actual_entsoe_transparency",
    "AT_load_forecast_entsoe_transparency",
    "AT_price_day_ahead",
    "AT_solar_generation_actual",
    "AT_wind_onshore_generation_actual"
]

df_at = df[at_columns].copy()

print("Vorschau der AT-Daten:")
print(df_at.head(), "\n")

# ---------------------------------
# Vollständigkeit
# ---------------------------------
# Überprüfung auf fehlende Werte
print("Fehlende Werte pro AT-Spalte:")
print(df_at.isnull().sum(), "\n")

# ---------------------------------
# Datumsrichtigkeit
# ---------------------------------
# Umwandlung der Zeitstempel in Datumsformt
df_at["utc_timestamp"] = pd.to_datetime(
    df_at["utc_timestamp"], errors="coerce"
)

df_at["cet_cest_timestamp"] = pd.to_datetime(
    df_at["cet_cest_timestamp"], errors="coerce"
)

# Überprüfung auf ungültige Zeitstempel
print("Ungültige UTC-Zeitstempel:")
print(df_at[df_at["utc_timestamp"].isnull()], "\n")

print("Ungültige CET/CEST-Zeitstempel:")
print(df_at[df_at["cet_cest_timestamp"].isnull()], "\n")

# ---------------------------------
# Wertebereiche
# ---------------------------------
# Überprüfung auf negative oder unrealistische Werte
print("Negative Lastwerte (AT_load_actual):")
print(df_at[df_at["AT_load_actual_entsoe_transparency"] < 0], "\n")

print("Negative Erzeugungswerte (Solar / Wind):")
print(
    df_at[
        (df_at["AT_solar_generation_actual"] < 0) |
        (df_at["AT_wind_onshore_generation_actual"] < 0)
    ],
    "\n"
)

# ---------------------------------
# Statistische Analyse
# ---------------------------------
# Wichtigsten numerischen Spalten
print("Deskriptive Statistik der AT-Stromdaten:")
print(
    df_at[
        [
            "AT_load_actual_entsoe_transparency",
            "AT_load_forecast_entsoe_transparency",
            "AT_price_day_ahead",
            "AT_solar_generation_actual",
            "AT_wind_onshore_generation_actual"
        ]
    ].describe(),
    "\n"
)

# ---------------------------------
# Analyse von Abweichungen
# ---------------------------------
# Differenz zwischen tatsächlicher und prognostizierter Last
df_at["load_difference"] = (
    df_at["AT_load_actual_entsoe_transparency"]
    - df_at["AT_load_forecast_entsoe_transparency"]
)

print("Abweichung zwischen Ist- und Prognoselast:")
print(df_at["load_difference"].describe(), "\n")

# ---------------------------------
# Korrelationen
# ---------------------------------
# Zusammenhängen zwischen Last, Preis und Erzeugung???
correlation_matrix = df_at[
    [
        "AT_load_actual_entsoe_transparency",
        "AT_price_day_ahead",
        "AT_solar_generation_actual",
        "AT_wind_onshore_generation_actual"
    ]
].corr()

print("Korrelationsmatrix (AT-Daten):")
print(correlation_matrix, "\n")
