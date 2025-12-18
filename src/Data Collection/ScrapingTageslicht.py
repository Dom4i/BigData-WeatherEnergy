import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://www.laenderdaten.info/Europa/Oesterreich/sonnenuntergang.php"

#Wir haben es nicht geschafft mit normalem Websracping, aus diesem Grund Manuell (es sind aber eh nur 12 Werte)
months = [
    "Jänner",
    "Februar",
    "März",
    "April",
    "Mai",
    "Juni",
    "Juli",
    "August",
    "September",
    "Oktober",
    "November",
    "Dezember"
]

day_lengths = ["8:52",
               "10:19",
               "11:56",
               "13:44",
               "15:16",
               "16:07",
               "15:44",
               "14:23",
               "12:39",
               "10:55",
               "9:18",
               "8:26"]

df = pd.DataFrame({"month": months, "day_length": day_lengths})
df.to_csv("data/raw/Tageslicht/sonnenlaenge_wien_monatlich.csv", index=False)
