import pandas as pd
from influxdb_client import InfluxDBClient

from pyspark.sql import SparkSession
from pyspark.sql.functions import hour, month, lag, lit
from pyspark.sql.window import Window

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, isnan, when, sum as spark_sum
import os
import sys

# %% [markdown]
# Daten aus InfluxDB laden in pandas DataFrame (weil influxDB DataFrames zurück gibt), feature engineering und Modell training mit Spark

# %%
# InfluxDB config analog zu docker-compose:
INFLUX_URL = "http://localhost:8086"
ORG = "bigdata"
BUCKET = "energy_weather"
TOKEN = "supersecrettoken"
MEASUREMENT = "features_hourly"

# Zeitintervall aller Daten: 2015-2019 (2020 exklusiv)
START = "2015-01-01T00:00:00Z"
STOP  = "2020-01-01T00:00:00Z"

# Flux Query: wandelt von long in wide table layout um, da influx normalerweise nur einen Wert pro Zeile speichert und für Machine Leaning sind feature vektoren besser (alle Einträge pro timestamp)
flux = f'''
from(bucket: "{BUCKET}")
  |> range(start: time(v: "{START}"), stop: time(v: "{STOP}"))
  |> filter(fn: (r) => r._measurement == "{MEASUREMENT}")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time",
    "load_mw","solar_mw","wind_mw",
    "temp_c","precip_mm","humidity_pct","wind_speed_kmh","pressure_hpa",
    "daylight_hours","daylight_seconds",
    "weekday","is_weekend","is_holiday"
  ])
  |> sort(columns: ["_time"])
'''

# Verbindung zu InfluxDB:
client = InfluxDBClient(url=INFLUX_URL, token=TOKEN, org=ORG)
q = client.query_api()

# Query ausführen: gibt pandas data frame zurück (pdf):
pdf = q.query_data_frame(flux)
pdf = pd.concat(pdf, ignore_index=True) if isinstance(pdf, list) else pdf # Falls mehrere DataFrames zurückkommen, zusammenführen
pdf = pdf.rename(columns={"_time": "timestamp"}) # _time → timestamp umbenennen

pdf["timestamp"] = pd.to_datetime(pdf["timestamp"], utc=True) # timestamps sicher als UTC datetime (pandas)

print("Loaded from Influx:", pdf.shape)
print(pdf.head(3))

# %%
# -------------------------
# Spark Session + Pandas -> Spark DF

# Spark hatte Probleme, Python interpreter zu finden. Folgende Systemvariablen sollen helfen:
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = SparkSession.builder.appName("InfluxSparkML").getOrCreate() # Spark session
df = spark.createDataFrame(pdf) # Pandas DF in Spark DF umwandeln


df = (df # InfluxDB Metadaten entferen (unnötig, stören bei ML)
      .drop("result", "table")
      .withColumn("timestamp", col("timestamp").cast("timestamp")) # timestamp als Spark timestamp casten
      .orderBy("timestamp")) # nach Zeit sortiern

# %%
# Feature Engineering (Zeitfeatures + Lags)
# Da das ursprüngliche Modell keine extrem hohe Genauigkeit hatte, wurden zusätzlich lags zum Modelltraining verwendet (das sind einfach Daten vom Energieverbruach der Vergangenheit)
# hour und month werden als features hinzugefügt, um Modellperformance zu verbessern

w = Window.orderBy("timestamp") # global nach timestamp sortieren

df_feat = (df
    .withColumn("hour", hour(col("timestamp")))
    .withColumn("month", month(col("timestamp")))
    .withColumn("load_lag_1", lag("load_mw", 1).over(w)) #1h vorher
    .withColumn("load_lag_24", lag("load_mw", 24).over(w)) #1 Tag vorher
    .withColumn("load_lag_168", lag("load_mw", 168).over(w)) #1 Woche vorher
).dropna(subset=["load_lag_1", "load_lag_24", "load_lag_168"]) # Erste Stunden haben keine Lags → entfernen (z.B. allererster Timestamp kann keine lags haben)

# %%
# Data Exploration

df_feat.printSchema() # Schema mit Datentypen prüfen

cols = [c for c in df_feat.columns if c != "timestamp"] # Alle Spalten außer timestamp -> timestamp hat error verursacht, kann irgendwie nicht auf nan Werte geprüft werden

# Fehlende Werte pro Spalte zählen:
missing_counts = df_feat.select([
    spark_sum(when(col(c).isNull() | isnan(col(c)), 1).otherwise(0)).alias(c)
    for c in cols
])
missing_counts.show(truncate=False)

# Folgende Spalten haben NaN werte, die Zeilen mit Nan entfernen (sind sehr wenige Zeilen, daher wird es wahrscheinlich keinen Bias verursachen):
df_clean = df_feat.dropna(subset=["pressure_hpa", "solar_mw", "wind_mw"])

# Kontrolle nach dem Cleaning:
cols_clean = [c for c in df_clean.columns if c != "timestamp"]
missing_counts_clean = df_clean.select([
    spark_sum(when(col(c).isNull() | isnan(col(c)), 1).otherwise(0)).alias(c)
    for c in cols_clean
])
missing_counts_clean.show(truncate=False)


# %%
# Chek, wie viele Zeilen gelöscht wurden:
#print("Before:", df_feat.count())
#print("After :", df_clean.count())

# Grundstatistiken der nummerischen Spalten:
df_clean.select(
    "load_mw","solar_mw","wind_mw",
    "temp_c","precip_mm","humidity_pct",
    "wind_speed_kmh","pressure_hpa"
).describe().show()

# %%

# Spark ML Pipeline (GBT Regressor)

# Trainingsfunktion: weil nun ein zweites Modell gemacht wird (einmal mit und einmal ohne lag):
def train_eval(df_train, df_test, feature_cols, label="load_mw"):
    assembler = VectorAssembler( # Features → Feature-Vektor
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="keep"
    )

    gbt = GBTRegressor( # Gradient Boosted Trees Regressor
        labelCol=label,
        featuresCol="features",
        maxIter=200,
        maxDepth=6
    )

    pipeline = Pipeline(stages=[assembler, gbt]) # Pipeline: Assembler → Modell
    model = pipeline.fit(df_train) # Modell trainieren
    pred = model.transform(df_test) # Vorhersagen für Testdaten

    # Evaluierung:
    rmse = RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName="rmse").evaluate(pred)
    mae  = RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName="mae").evaluate(pred)
    return model, pred, rmse, mae

feature_cols = ["solar_mw", "wind_mw", "temp_c", "precip_mm", "humidity_pct", "wind_speed_kmh", "pressure_hpa", "daylight_hours", "daylight_seconds", "weekday", "is_weekend", "is_holiday", "hour", "month", "load_lag_1", "load_lag_24", "load_lag_168",]

# Feature sets für beide Modelle:
feature_cols_with_lags = feature_cols
feature_cols_no_lags = [c for c in feature_cols if not c.startswith("load_lag_")]

# Train/Test split für Modellperformance (macht man bei Zeitreihen normalerweise nicht random sondern auf "zukünftige" Daten:
train = df_clean.filter(col("timestamp") < lit("2019-01-01 00:00:00"))
test  = df_clean.filter(col("timestamp") >= lit("2019-01-01 00:00:00"))

# Training + evaluation durch Funktion:
model_lags, pred_lags, rmse_lags, mae_lags = train_eval(train, test, feature_cols_with_lags)
model_nolag, pred_nolag, rmse_nolag, mae_nolag = train_eval(train, test, feature_cols_no_lags)

print("\nVergleich")
print(f"MIT  Lags -> RMSE: {rmse_lags:.3f} | MAE: {mae_lags:.3f}")
print(f"OHNE Lags -> RMSE: {rmse_nolag:.3f} | MAE: {mae_nolag:.3f}")

# Feature Importances (mit Lags):
gbt_lags = model_lags.stages[1]
imp_lags = pd.DataFrame({
    "feature": feature_cols_with_lags,
    "importance": gbt_lags.featureImportances.toArray()
}).sort_values("importance", ascending=False)

print("\nFeature Importances (MIT Lags)")
print(imp_lags)

# Feature Importances (ohne Lags):
gbt_nolag = model_nolag.stages[1]
imp_nolag = pd.DataFrame({
    "feature": feature_cols_no_lags,
    "importance": gbt_nolag.featureImportances.toArray()
}).sort_values("importance", ascending=False)

print("\nFeature Importances (OHNE Lags)")
print(imp_nolag)