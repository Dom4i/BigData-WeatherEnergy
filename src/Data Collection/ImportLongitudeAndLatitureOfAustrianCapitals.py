from geopy.geocoders import Nominatim
import time
import json

# Landeshauptstädte Österreichs
CITIES = [
    "Bregenz",
    "Innsbruck",
    "Salzburg",
    "Linz",
    "Wien",
    "Graz",
    "Klagenfurt"
]

OUTPUT_FILE = "austria_places_latlon.json"

geolocator = Nominatim(user_agent="austria-daylight-analysis")

def geocode_city(city: str):
    location = geolocator.geocode(f"{city}, Austria")
    if location is None:
        raise ValueError(f"Ort nicht gefunden: {city}")
    return {
        "latitude": round(location.latitude, 6),
        "longitude": round(location.longitude, 6)
    }

places = {}

for city in CITIES:
    print(f"Geocoding {city} ...")
    places[city] = geocode_city(city)
    time.sleep(1)  # Rate-Limit kann überschritten werden, daher wait

# Speichern als JSON
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(places, f, indent=2, ensure_ascii=False)

print(f"\nFertig! Koordinaten gespeichert in: {OUTPUT_FILE}")