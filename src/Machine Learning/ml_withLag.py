# train_from_influx_spark.py
# InfluxDB (Flux) -> Pandas -> Spark DF -> Feature Engineering -> Spark ML (GBT)

import pandas as pd
from influxdb_client import InfluxDBClient

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, month, lag, lit
from pyspark.sql.window import Window

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, isnan, when, sum as spark_sum
import os
import sys

# -------------------------
# 1) InfluxDB config (aus deinem docker-compose)
# -------------------------
INFLUX_URL = "http://localhost:8086"
ORG = "bigdata"
BUCKET = "energy_weather"
TOKEN = "supersecrettoken"
MEASUREMENT = "features_hourly"

START = "2015-01-01T00:00:00Z"
STOP  = "2020-01-01T00:00:00Z"   # exklusiv, damit 2019 komplett drin ist

# -------------------------
# 2) Flux Query: measurement -> wide table (pivot)
# -------------------------
# Falls Field-Namen anders heißen, unten in keep(columns: [...]) anpassen.
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

client = InfluxDBClient(url=INFLUX_URL, token=TOKEN, org=ORG)
q = client.query_api()

pdf = q.query_data_frame(flux)
pdf = pd.concat(pdf, ignore_index=True) if isinstance(pdf, list) else pdf
pdf = pdf.rename(columns={"_time": "timestamp"})

# timestamps sicher als UTC datetime (pandas)
pdf["timestamp"] = pd.to_datetime(pdf["timestamp"], utc=True)

print("Loaded from Influx:", pdf.shape)
print(pdf.head(3))

# -------------------------
# 3) Spark Session + Pandas -> Spark DF
# -------------------------

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
spark = SparkSession.builder.appName("InfluxSparkML").getOrCreate()

df = spark.createDataFrame(pdf)

# Spark timestamp cast (UTC-nahe; Spark speichert intern ohne TZ, aber wir nutzen UTC Strings)
df = (df
      .drop("result", "table")
      .withColumn("timestamp", col("timestamp").cast("timestamp"))
      .orderBy("timestamp"))

# Optional: solar nachts als 0 statt NaN (wenn du das physikalisch willst)
# df = df.fillna({"solar_mw": 0.0})

# %%
# -------------------------
# 4) Feature Engineering (Zeitfeatures + Lags)
# -------------------------
w = Window.orderBy("timestamp")

df_feat = (df
    .withColumn("hour", hour(col("timestamp")))
    .withColumn("month", month(col("timestamp")))
    .withColumn("load_lag_1",   lag("load_mw", 1).over(w))
    .withColumn("load_lag_24",  lag("load_mw", 24).over(w))
    .withColumn("load_lag_168", lag("load_mw", 168).over(w))
).dropna(subset=["load_lag_1", "load_lag_24", "load_lag_168"])

# %%
# Data Exploration

df_feat.printSchema()

cols = [c for c in df_feat.columns if c != "timestamp"]

missing_counts = df_feat.select([
    spark_sum(when(col(c).isNull() | isnan(col(c)), 1).otherwise(0)).alias(c)
    for c in cols
])
missing_counts.show(truncate=False)

df_clean = df_feat.dropna(subset=["pressure_hpa", "solar_mw", "wind_mw"])

cols_clean = [c for c in df_clean.columns if c != "timestamp"]
missing_counts_clean = df_clean.select([
    spark_sum(when(col(c).isNull() | isnan(col(c)), 1).otherwise(0)).alias(c)
    for c in cols_clean
])
missing_counts_clean.show(truncate=False)


# %%
#print("Before:", df_feat.count())
#print("After :", df_clean.count())

df_clean.select(
    "load_mw","solar_mw","wind_mw",
    "temp_c","precip_mm","humidity_pct",
    "wind_speed_kmh","pressure_hpa"
).describe().show()
# %%
# -------------------------
# 5) Zeitbasierter Split
# -------------------------
train = df_clean.filter(col("timestamp") < lit("2019-01-01 00:00:00"))
test  = df_clean.filter(col("timestamp") >= lit("2019-01-01 00:00:00"))

print("Train rows:", train.count())
print("Test rows :", test.count())

# -------------------------
# 6) Spark ML Pipeline (GBT Regressor)
# -------------------------
label = "load_mw"

feature_cols = [
    # Energie + Wetter + Tageslicht + Kalender
    "solar_mw", "wind_mw",
    "temp_c", "precip_mm", "humidity_pct", "wind_speed_kmh", "pressure_hpa",
    "daylight_hours", "daylight_seconds",
    "weekday", "is_weekend", "is_holiday",
    # Zeitfeatures
    "hour", "month",
    # Lags (sehr wichtig)
    "load_lag_1", "load_lag_24", "load_lag_168",
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="keep"  # falls irgendwo noch NaNs bleiben
)

gbt = GBTRegressor(
    labelCol=label,
    featuresCol="features",
    maxIter=200,
    maxDepth=6
)

pipeline = Pipeline(stages=[assembler, gbt])
model = pipeline.fit(train)

pred = model.transform(test)

# -------------------------
# 7) Evaluation
# -------------------------
rmse = RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName="rmse").evaluate(pred)
mae  = RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName="mae").evaluate(pred)

print("RMSE:", rmse)
print("MAE :", mae)

# Optional: ein paar Predictions anschauen
pred.select("timestamp", "load_mw", "prediction").orderBy("timestamp").show(10, truncate=False)

# %%

gbt_model = model.stages[1]
importances = gbt_model.featureImportances
feat_imp = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances.toArray()
})

feat_imp = feat_imp.sort_values("importance", ascending=False)
print(feat_imp)