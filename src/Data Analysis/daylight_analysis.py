import pandas as pd

# =========================================================
# Analyse der monatlichen Sonnenlänge in Wien
# =========================================================

file_path = "../../data/raw/Tageslicht/sonnenlaenge_wien_monatlich.csv"

df = pd.read_csv(file_path)

print("Vorschau der Rohdaten:")
print(df, "\n")

# ---------------------------------
# Vollständigkeit
# ---------------------------------
print("Fehlende Werte pro Spalte:")
print(df.isnull().sum(), "\n")

# Überprüfung, ob alle 12 Monate vorhanden sind
print("Anzahl der Monate im Datensatz:")
print(len(df), "\n")

# ---------------------------------
# Richtigkeit des Zeitformats
# ---------------------------------
# Auftrennen der Tageslänge in hh:mm
time_split = df["day_length"].str.split(":", expand=True)

df["hours"] = pd.to_numeric(time_split[0], errors="coerce")
df["minutes"] = pd.to_numeric(time_split[1], errors="coerce")

# Umrechnung der Tageslänge in Minuten
df["day_length_minutes"] = df["hours"] * 60 + df["minutes"]

print("Ungültige Zeitangaben:")
print(df[df["day_length_minutes"].isnull()], "\n")

# Plausibilität?? (0–1440 Minuten)
print("Unplausible Tageslängen:")
print(df[(df["day_length_minutes"] < 0) | (df["day_length_minutes"] > 1440)], "\n")

# ---------------------------------
# Statistische Analyse
# ---------------------------------
print("Deskriptive Statistik der Tageslänge (Minuten):")
print(df["day_length_minutes"].describe(), "\n")

# Monat mit längstem und kürzestem Tag
max_day = df.loc[df["day_length_minutes"].idxmax()]
min_day = df.loc[df["day_length_minutes"].idxmin()]

print("Monat mit längster Tageslänge:")
print(max_day[["month", "day_length"]], "\n")

print("Monat mit kürzester Tageslänge:")
print(min_day[["month", "day_length"]], "\n")

# ---------------------------------
# Abhängigkeit
# ---------------------------------

df["month_index"] = range(1, len(df) + 1)

correlation = df["month_index"].corr(df["day_length_minutes"])
print("Korrelation zwischen Monatsindex und Tageslänge:")
print(correlation, "\n")
