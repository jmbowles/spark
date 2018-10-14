"""
https://github.com/apache/spark/blob/master/examples/src/main/python/sql/arrow.py
https://towardsdatascience.com/outlier-detection-with-isolation-forest-3d190448d45e
https://databricks.com/blog/2017/10/30/introducing-vectorized-udfs-for-pyspark.html
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
"""

from __future__ import print_function

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.utils import require_minimum_pandas_version, require_minimum_pyarrow_version
from pyspark.sql.functions import lit, col, rand, pandas_udf, PandasUDFType

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


require_minimum_pandas_version()
require_minimum_pyarrow_version()

# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "true")


# Generate fake data that groups source_ip and their respective sum_bytes
df = spark.range(0, 10 * 10).withColumnRenamed("id", "source_ip").withColumn("source_ip", (col("source_ip") / 1).cast("integer"))
df = df.withColumn("sum_bytes", (rand(seed=42)*25562).cast("integer"))
df = df.withColumn("count_url", (rand(seed=42)*32).cast("integer"))
df = df.withColumn("predicted", lit(0))

# Extract the schema for the pandas udf
schema = df.schema

@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def isolation_forest(pdf):
	rng = np.random.RandomState(42)
	forest = IsolationForest(max_samples=100, random_state=rng)
	x = pd.DataFrame([pdf.sum_bytes, pdf.count_url])
	y = forest.fit_predict(x.T)
	return pd.DataFrame({"source_ip": pdf.source_ip, "sum_bytes": pdf.sum_bytes, "count_url": pdf.count_url, "predicted": y}).reindex(columns=["source_ip", "sum_bytes", "count_url", "predicted"])

results = df.groupby("source_ip").apply(isolation_forest)
results.show()