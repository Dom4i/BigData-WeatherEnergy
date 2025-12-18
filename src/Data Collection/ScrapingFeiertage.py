import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "https://www.ferienwiki.at/feiertage/{}/at"

data = []

for year in range(2015, 2021):
    print(f"Lade Daten für {year}...")
    url = BASE_URL.format(year)

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.find("table", class_="table")
    tbody = table.find("tbody")

    for row in tbody.find_all("tr"):
        cols = row.find_all("td")

        if len(cols) != 4:
            continue

        name = cols[0].get_text(strip=True)
        if not name:
            name = "Unbekannt"

        date_text = cols[1].get_text(strip=True)
        if not date_text:
            continue

        data.append({
            "year": year,
            "holiday_name": name,
            "date_raw": date_text
        })

df = pd.DataFrame(data)
print(df)

df.to_csv("data/raw/Feiertage/feiertage_at_2015_2020_raw.csv", index=False)


# Verbessern der CSV

df = pd.read_csv("../../data/raw/Feiertage/feiertage_at_2015_2020_raw.csv")

df["date_str"] = df["date_raw"].str.extract(r"(\d{2}\.\d{2}\.\d{4})")
df["weekday"] = df["date_raw"].str.extract(r"\((.*?)\)")

df["date"] = pd.to_datetime(df["date_str"], format="%d.%m.%Y")

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

df_clean = df[[
    "year",
    "month",
    "day",
    "date",
    "weekday",
    "holiday_name"
]]

df_clean.to_csv("data/raw/Feiertage/feiertage_at_2015_2020_clean.csv", index=False)

print("Neues sauberes CSV wurde erstellt")

