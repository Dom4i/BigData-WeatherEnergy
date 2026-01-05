from datetime import date, datetime, timedelta, time
from zoneinfo import ZoneInfo
from astral import LocationInfo
from astral.sun import sun
import csv
import json
import statistics

TZ = ZoneInfo("Europe/Vienna") # Zeitzone definieren

# Koordinaten der Landeshauptstädte aus JSON laden:
with open("austria_places_latlon.json", encoding="utf-8") as f:
    PLACES = json.load(f)

# Wandelt einen Zeitpunkt (datetime) in Sekunden seit Mitternacht um:
def seconds_since_midnight(dt: datetime) -> int:
    #dt ist timezone-aware. Liefert Sekunden seit lokalem Mitternacht. Warum Sekunden seit Mitternacht? Weil durchscnitt über Landeshauptstädte von Österreich berechnet wird
    return dt.hour * 3600 + dt.minute * 60 + dt.second

# Umkehrung von Funktion darüber: wandelt von Sekunden seit Mitternacht in Zeitformat um
def hhmmss_from_seconds(sec: int) -> str:
    sec = int(round(sec))
    sec = sec % 86400
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# berechnet Sonnenauf- und Untergang für einen Ort + Datum
def sunrise_sunset_for_place(lat: float, lon: float, d: date) -> tuple[datetime, datetime]:
    loc = LocationInfo("x", "AT", "Europe/Vienna", lat, lon)
    s = sun(loc.observer, date=d, tzinfo=TZ)
    return s["sunrise"], s["sunset"]

# Berechnet durchschnittliche Sonnenaufgangzeit und Sonnenuntergrangzeit für ganz Österreich (Mittelwert der Landeshauptstädte)
def austria_avg_sunrise_sunset(d: date) -> tuple[int, int]:
    sunrise_secs = []
    sunset_secs = []

    for coords in PLACES.values():
        sr, ss = sunrise_sunset_for_place(coords["latitude"], coords["longitude"], d)
        sunrise_secs.append(seconds_since_midnight(sr))
        sunset_secs.append(seconds_since_midnight(ss))

    return int(round(statistics.mean(sunrise_secs))), int(round(statistics.mean(sunset_secs)))

# Datumsbereich:
def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)

# Letzten 5 Jahre
end = date.today()
start = date(end.year - 5, end.month, end.day)

# Ausgabe speichern: pro Tag durchschnittliche Sonnenauf- und Untergrang Zeitpunkte
# Sekunden seit Mitternacht werden auch gespeichert falls sie im Projekt benötigt werden
out_file = "austria_sunrise_sunset_avg_last5y_daily.csv"
with open(out_file, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["date", "sunrise_avg", "sunset_avg", "sunrise_avg_seconds", "sunset_avg_seconds"])

    for d in daterange(start, end):
        sr_sec, ss_sec = austria_avg_sunrise_sunset(d)
        w.writerow([
            d.isoformat(),
            hhmmss_from_seconds(sr_sec),
            hhmmss_from_seconds(ss_sec),
            sr_sec,
            ss_sec
        ])

print(f"Fertig! CSV gespeichert: {out_file}")