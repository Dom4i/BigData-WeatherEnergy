from datetime import date, timedelta
from zoneinfo import ZoneInfo
from astral import LocationInfo
from astral.sun import sun
import csv
import statistics
import json

TZ = ZoneInfo("Europe/Vienna")

with open("austria_places_latlon.json", encoding="utf-8") as f:
    PLACES = json.load(f)  # {"Wien": {"latitude": ..., "longitude": ...}, ...}

def daylight_hours(lat: float, lon: float, d: date) -> float:
    loc = LocationInfo("x", "AT", "Europe/Vienna", lat, lon)
    s = sun(loc.observer, date=d, tzinfo=TZ)
    return (s["sunset"] - s["sunrise"]).total_seconds() / 3600.0

def austria_avg_daylight(d: date) -> float:
    vals = [
        daylight_hours(coords["latitude"], coords["longitude"], d)
        for coords in PLACES.values()
    ]
    return statistics.mean(vals)

def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)

# Letzten 5 Jahre importieren -> Check, ob wir das brauchen (je nachdem wie lange die Energieverbrauch Daten zurück gehen)
end = date.today()
start = date(end.year - 5, end.month, end.day)

out_file = "austria_daylight_last5y_daily.csv"
with open(out_file, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["date", "daylight_hours_avg_AT"])
    for d in daterange(start, end):
        h = austria_avg_daylight(d)
        w.writerow([d.isoformat(), f"{h:.6f}"])

print(f"Fertig! CSV gespeichert: {out_file}")