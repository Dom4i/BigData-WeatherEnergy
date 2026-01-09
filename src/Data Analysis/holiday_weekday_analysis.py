import pandas as pd

# =========================================================
# Analyse von Feiertagen und Wochentagen
# ist aber Webscraping also sollte alles passen
# und ich muss sagen ich hab bisschen was unnötiges da drinnen auch gemacht haha
# =========================================================

# Pfad
file_path = "../../data/raw/Feiertage/feiertage_at_2015_2020_raw.csv"

# CSV-Datei einlesen
df = pd.read_csv(file_path)

print("Vorschau der Rohdaten:")
print(df.head(), "\n")

# ---------------------------------
# Vollständigkeit
# ---------------------------------
# fehlende Werte pro Spalte
print("Fehlende Werte pro Spalte:")
print(df.isnull().sum(), "\n")

# ---------------------------------
# Trennung von Datum und Wochentag
# ---------------------------------
# Extraktion des Datums in TT.MM.YYYY
df["date_str"] = df["date_raw"].str.extract(r"(\d{2}\.\d{2}\.\d{4})")

df["weekday_text"] = df["date_raw"].str.extract(r"\((.*?)\)")

# Umwandlung in ein echtes Datumsformat
df["date"] = pd.to_datetime(
    df["date_str"],
    format="%d.%m.%Y",
    errors="coerce"
)

# Ausgabe von Zeilen mit ungültigen Wert
print("Zeilen mit ungültigen Datumswerten:")
print(df[df["date"].isnull()], "\n")

# ---------------------------------
# Berechnung des Wochentags aus dem Datum (schauen ob das stimmt)
# ---------------------------------
# Mapping englischen auf deutsche
weekday_map = {
    "Monday": "Montag",
    "Tuesday": "Dienstag",
    "Wednesday": "Mittwoch",
    "Thursday": "Donnerstag",
    "Friday": "Freitag",
    "Saturday": "Samstag",
    "Sunday": "Sonntag"
}

# Berechnung des Wochentags aus Datum
df["weekday_calc"] = df["date"].dt.day_name().map(weekday_map)

# Vergleich zwischen angegebenem und berechnetem Wochentag
df["weekday_match"] = df["weekday_text"] == df["weekday_calc"]

print("Anzahl inkorrekter Wochentagsangaben:")
print((~df["weekday_match"]).sum(), "\n")

# ---------------------------------
# Statistische Analyse
# ---------------------------------

# Anzahl der Feiertage pro Jahr
holidays_per_year = df.groupby("year").size()
print("Anzahl der Feiertage pro Jahr:")
print(holidays_per_year, "\n")

# Verteilung der Feiertage nach Wochentagen
weekday_distribution = df["weekday_calc"].value_counts()
print("Verteilung der Feiertage nach Wochentagen:")
print(weekday_distribution, "\n")

# Häufigste Feiertage
most_common_holidays = df["holiday_name"].value_counts().head(10)
print("Häufigste Feiertage:")
print(most_common_holidays, "\n")

# ---------------------------------
# Zusammenfassende Statistik
# ---------------------------------

print("Zusammenfassende Statistik:")
print(holidays_per_year.describe(), "\n")
