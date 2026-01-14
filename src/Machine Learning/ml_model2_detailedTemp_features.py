# ml_model2_detailedTemp_features.py
# Parquet (Pandas/PyArrow) -> Spark DF -> Feature Engineering -> Spark ML (GBT)
# Lags: 1h, 24h, 168h
# Fix: Spark kann TIMESTAMP(NANOS,true) oft nicht direkt lesen -> daher via Pandas
# Fix: NaN/Inf in Features vor Training entfernen

import os
import sys
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, month, lag, lit, isnan
from pyspark.sql.window import Window

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator



# Spark Setup
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = SparkSession.builder.appName("Model2_DetailedTemps_3Lags").getOrCreate()


# Parquet laden (RELATIV, via Pandas wegen TIMESTAMP(NANOS,true))
PARQUET_PATH = "../../data/processed/features/features_hourly_AT_2015_2019_model2.parquet"

pdf = pd.read_parquet(PARQUET_PATH, engine="pyarrow")

# Timestamp-Spalte vereinheitlichen
if "_time" in pdf.columns and "timestamp" not in pdf.columns:
    pdf = pdf.rename(columns={"_time": "timestamp"})

if "timestamp" not in pdf.columns:
    raise ValueError(f"Keine Timestamp-Spalte gefunden. Spalten: {list(pdf.columns)[:50]} ...")

# Spark-kompatibel machen:
# - tz-aware -> tz entfernen (UTC -> naive)
# - auf microseconds casten (Spark-freundlicher)
pdf["timestamp"] = pd.to_datetime(pdf["timestamp"], utc=True).dt.tz_convert(None)
pdf["timestamp"] = pdf["timestamp"].astype("datetime64[us]")

# Pandas -> Spark DataFrame
df = spark.createDataFrame(pdf)

# Ordnung / Cast
df = (
    df.withColumn("timestamp", col("timestamp").cast("timestamp"))
      .orderBy("timestamp")
)

# Feature Engineering (Zeit + Lags)

label = "load_mw"
if label not in df.columns:
    raise ValueError(f"Label-Spalte '{label}' fehlt im Parquet!")

w = Window.orderBy("timestamp")

df_feat = (
    df
    .withColumn("hour", hour(col("timestamp")))
    .withColumn("month", month(col("timestamp")))
    #.withColumn("load_lag_1",   lag(label, 1).over(w))
    .withColumn("load_lag_24",  lag(label, 24).over(w))
    .withColumn("load_lag_168", lag(label, 168).over(w))
    .dropna(subset=[label, "load_lag_24", "load_lag_168"])
)


# Feature Auswahl
# Detaillierte Temperaturen: temp_c_*
temp_cols = sorted([c for c in df_feat.columns if c.startswith("temp_c_")])

# Baseline Wetter / Kalender
optional_base = [
    "solar_mw", "wind_mw",
    "temp_c", "precip_mm", "humidity_pct",
    "wind_speed_kmh", "pressure_hpa",
    "daylight_hours", "daylight_seconds",
    "weekday", "is_weekend", "is_holiday",
]
base_cols = [c for c in optional_base if c in df_feat.columns]

feature_cols = (
    base_cols
    + temp_cols
    + ["hour", "month"]
    + ["load_lag_24", "load_lag_168"]
)
feature_cols = [c for c in feature_cols if c != label]

print(f"Using {len(feature_cols)} features.")
print("First 30 feature cols:", feature_cols[:30])

if len(feature_cols) == 0:
    raise ValueError("Keine Feature-Spalten gefunden (feature_cols ist leer).")


# NaN/Inf Cleanup

# NaNs in Feature-Spalten rausfiltern
nan_expr = None
for c in feature_cols:
    if c in df_feat.columns:
        cond = isnan(col(c))
        nan_expr = cond if nan_expr is None else (nan_expr | cond)

if nan_expr is not None:
    df_feat = df_feat.filter(~nan_expr)

# +/-Infinity rausfiltern (Spark hat dafür kein isnan)
for c in feature_cols:
    if c in df_feat.columns:
        df_feat = df_feat.filter(
            (col(c).isNull()) | ((col(c) != float("inf")) & (col(c) != float("-inf")))
        )

# Nulls in Features + Label entfernen
df_feat = df_feat.dropna(subset=feature_cols + [label])


# Zeitbasierter Split
train = df_feat.filter(col("timestamp") < lit("2019-01-01 00:00:00"))
test  = df_feat.filter(col("timestamp") >= lit("2019-01-01 00:00:00"))

print("Train rows:", train.count())
print("Test rows :", test.count())


# ML Pipeline
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="keep"
)

gbt = GBTRegressor(
    labelCol=label,
    featuresCol="features",
    maxIter=200,
    maxDepth=6
)

model = Pipeline(stages=[assembler, gbt]).fit(train)
pred = model.transform(test)


# Evaluation
rmse = RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName="rmse").evaluate(pred)
mae  = RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName="mae").evaluate(pred)

print("RMSE:", rmse)
print("MAE :", mae)

pred.select("timestamp", "load_mw", "prediction").orderBy("timestamp").show(10, truncate=False)

# Feature Importances
gbt_model = model.stages[1]
importances = gbt_model.featureImportances.toArray()

feat_imp = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nTop 30 Feature Importances:")
print(feat_imp.head(30).to_string(index=False))

spark.stop()
