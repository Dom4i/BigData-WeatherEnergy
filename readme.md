# BigData-WeatherEnergy

Analyse des österreichischen Energieverbrauchs in Abhängigkeit von Wetter- und Kalendereinflüssen.

---

## Python Setup

### Virtuelle Umgebung aktivieren
```bash
.\.venv\Scripts\Activate

pip install -r requirements.txt
```
### requirements.txt aktualisieren
```bash
pip freeze > requirements.txt
```
## Business Info
Stromverbrauch wird stündlich als Mittelwert gespeichert
z.B. 17:00 50000 MW verbraucht bedeutet, dass zwischen 17:00 und 18:00 Uhr im Schnitt 5000 MW verbraucht werden