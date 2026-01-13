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
import numpy as np
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
os.environ["JAVA_TOOL_OPTIONS"] = "-Dlog4j2.level=ERROR" # keine Warnings ausgeben

spark = SparkSession.builder.appName("InfluxSparkML").getOrCreate() # Spark session
spark.sparkContext.setLogLevel("ERROR") # keine Warnings ausgeben
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

# Folgende Spalten haben NaN Werte, die Zeilen mit Nan entfernen (sind sehr wenige Zeilen, daher wird es wahrscheinlich keinen Bias verursachen):
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
print("Before:", df_feat.count())
print("After :", df_clean.count())

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

# %%
# Drittes Modell ohne lag_1, damit man Daten mit den Vorhersagen der Energieanbieter vergleichen kann
feature_cols_lags_24_168 = [
    c for c in feature_cols if c not in ("load_lag_1",)
]

model_lags_24_168, pred_lags_24_168, rmse_lags_24_168, mae_lags_24_168 = train_eval(
    train, test, feature_cols_lags_24_168
)

gbt_cols_lags_24_168 = model_lags_24_168.stages[1]
imp_cols_lags_24_168 = pd.DataFrame({
    "feature": feature_cols_lags_24_168,
    "importance": gbt_cols_lags_24_168.featureImportances.toArray()
}).sort_values("importance", ascending=False)

print("\nZusätzliches Modell")
print(f"MIT Lags (24h & 168h) -> RMSE: {rmse_lags_24_168:.3f} | MAE: {mae_lags_24_168:.3f}")

# %%
# Daten von Energieanbieter evaluieren: vielleicht doch wieder löschen oder nicht in notebook geben?

CSV_PATH = "data/raw/Strom/time_series_60min_singleindex.csv"

# 1) Forecast laden
pdf_forecast = pd.read_csv(
    CSV_PATH,
    usecols=["utc_timestamp", "AT_load_forecast_entsoe_transparency"]
).rename(columns={
    "utc_timestamp": "timestamp",
    "AT_load_forecast_entsoe_transparency": "entsoe_forecast_mw"
})

pdf_forecast["timestamp"] = pd.to_datetime(pdf_forecast["timestamp"], utc=True)

# 2) Deine echten load-Daten (Spark -> Pandas), aber nur nötige Spalten & nur 2019
pdf_load_2019 = (
    df_clean
    .select("timestamp", "load_mw")
    .filter(col("timestamp") >= lit("2019-01-01 00:00:00"))
    .toPandas()
)

# sicherstellen, dass timestamp auch UTC-aware ist
pdf_load_2019["timestamp"] = pd.to_datetime(pdf_load_2019["timestamp"], utc=True)

# 3) Join in Pandas
pdf_base = pdf_load_2019.merge(pdf_forecast, on="timestamp", how="inner").dropna()

# 4) Metriken
err = (pdf_base["entsoe_forecast_mw"] - pdf_base["load_mw"]).to_numpy()

baseline_mae = np.mean(np.abs(err))
baseline_rmse = np.sqrt(np.mean(err**2))
baseline_rmae = baseline_mae / pdf_base["load_mw"].mean()  # relative MAE

print("\nENTSO-E Forecast Baseline (Test 2019) [Pandas]")
print(f"Rows used: {len(pdf_base)}")
print(f"MAE  : {baseline_mae:.3f}")
print(f"RMSE : {baseline_rmse:.3f}")
print(f"RMAE : {baseline_rmae:.4f}")

# %%
# Plots:

# %%
import matplotlib.pyplot as plt

def pred_to_pandas(pred_spark, label_col="load_mw", time_col="timestamp", n_max=None):
    sdf = pred_spark.select(time_col, col(label_col).alias("y"), col("prediction").alias("yhat")).orderBy(time_col)
    if n_max is not None:
        sdf = sdf.limit(int(n_max))
    pdfp = sdf.toPandas()
    pdfp[time_col] = pd.to_datetime(pdfp[time_col], utc=True)
    pdfp["residual"] = pdfp["yhat"] - pdfp["y"]
    return pdfp

def plot_scatter_actual_vs_pred(pdf_pred, title):
    # scatterplot predicted vs. actual
    d = pdf_pred.dropna()
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(d["y"], d["yhat"], s=8, alpha=0.35)
    mn = float(min(d["y"].min(), d["yhat"].min()))
    mx = float(max(d["y"].max(), d["yhat"].max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.title(title)
    plt.xlabel("Actual (MW)")
    plt.ylabel("Predicted (MW)")
    plt.tight_layout()
    plt.show()

def plot_feature_importance(imp_df, title, top_n=15):
    # Balkendiagramm mit feature importance
    d = imp_df.sort_values("importance", ascending=False).head(top_n)
    plt.figure(figsize=(8, 4.5))
    plt.barh(d["feature"][::-1], d["importance"][::-1])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

def plot_metrics_bar(metrics_rows, title="Model Metrics (Test 2019)"):
    # Balkendiagramm RMSE und MAE pro Modell

    mdf = pd.DataFrame(metrics_rows)
    x = np.arange(len(mdf))

    plt.figure(figsize=(8, 4))
    plt.bar(x - 0.2, mdf["rmse"], width=0.4, label="RMSE")
    plt.bar(x + 0.2, mdf["mae"],  width=0.4, label="MAE")
    plt.xticks(x, mdf["model"], rotation=15, ha="right")
    plt.title(title)
    plt.ylabel("Error (MW)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# %%

pdf_pred_lags = pred_to_pandas(pred_lags)
pdf_pred_nolag = pred_to_pandas(pred_nolag)
pdf_pred_lags_24_168 = pred_to_pandas(pred_lags_24_168)

plot_scatter_actual_vs_pred(pdf_pred_lags, "Scatter: Actual vs Predicted (MIT Lags)")
plot_scatter_actual_vs_pred(pdf_pred_nolag, "Scatter: Actual vs Predicted (OHNE Lags)")
plot_scatter_actual_vs_pred(pdf_pred_lags_24_168, "Scatter: Actual vs Predicted (Lags 24h & 168h)")

# %%
plot_feature_importance(imp_lags, "Feature Importances (MIT Lags) – Top 15", top_n=15)
plot_feature_importance(imp_nolag, "Feature Importances (OHNE Lags) – Top 15", top_n=15)
plot_feature_importance(imp_cols_lags_24_168, "Feature Importances (mit 24 und 169 lags, ohne 1) – Top 15", top_n=15)

# %%

# %%
metrics_rows = [
    {"model": "GBT lags (1,24,168)", "rmse": rmse_lags, "mae": mae_lags},
    {"model": "GBT no lags", "rmse": rmse_nolag, "mae": mae_nolag},
    {"model": "GBT lags (24,168)", "rmse": rmse_lags_24_168, "mae": mae_lags_24_168},
]
plot_metrics_bar(metrics_rows, title="Vergleich: RMSE & MAE (Test 2019)")