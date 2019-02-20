"""
https://github.com/apache/spark/blob/master/examples/src/main/python/sql/arrow.py
https://towardsdatascience.com/outlier-detection-with-isolation-forest-3d190448d45e
https://databricks.com/blog/2017/10/30/introducing-vectorized-udfs-for-pyspark.html
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/iforest.py

"""

from __future__ import print_function
from __future__ import division

from pyspark.sql import SparkSession
from pyspark.sql.utils import require_minimum_pandas_version, require_minimum_pyarrow_version
from pyspark.sql.functions import lit, col, rand, pandas_udf, PandasUDFType, approx_count_distinct
from pyspark.sql.types import StructType, StructField, IntegerType

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


require_minimum_pandas_version()
require_minimum_pyarrow_version()

# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "true")


# Generate fake data that groups source_ip and their respective sum_bytes
df = spark.range(0, 100 * 100).withColumnRenamed("id", "source_ip").withColumn("source_ip", (col("source_ip") / 1).cast("integer"))
df = df.withColumn("sum_bytes", (rand(seed=42)*25562).cast("integer"))
df = df.withColumn("count_url", (rand(seed=42)*32).cast("integer"))

# Specify the schema for the return dataframe. Input dataframe and output dataframes can be different since using GROUPED_MAP
schema = StructType([StructField("source_ip", IntegerType(), False), StructField("sum_bytes", IntegerType(), False), StructField("count_url", IntegerType(), False), StructField("prediction", IntegerType(), False)])

@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def isolation_forest(pdf):
	# Only include columns to be fit. By default, apply function will include column that was grouped upon
	features = pdf[["sum_bytes", "count_url"]]
	max_features = features.shape[1]
	forest = IsolationForest(n_jobs=4, behaviour=None, n_estimators=300, contamination="auto", max_samples=100000, max_features=max_features)
	predictions = forest.fit_predict(features)
	return pd.DataFrame({"source_ip": pdf.source_ip, "sum_bytes": features.sum_bytes, "count_url": features.count_url, "prediction": predictions})

results = df.groupby("source_ip").apply(isolation_forest)
results.show()
#results.groupby("prediction").agg(approx_count_distinct(results.prediction)).show()


rng = np.random.RandomState(42)

# Generating training data 
X_train = 0.2 * rng.randn(10000, 2)
X_train = np.r_[X_train + 3, X_train]
X_train = pd.DataFrame(X_train, columns = ["x1", "x2"])

max_samples = X_train.shape[0] - 1
max_features = X_train.shape[1]

# Generating new, 'normal' observation
X_test = 0.2 * rng.randn(2000, 2)
X_test = np.r_[X_test + 3, X_test]
X_test = pd.DataFrame(X_test, columns = ["x1", "x2"])

# Generating outliers
X_outliers = rng.uniform(low=-1, high=5, size=(1000, 2))
X_outliers = pd.DataFrame(X_outliers, columns = ["x1", "x2"])

# training the model
clf = IsolationForest(n_jobs=4, behaviour=None, n_estimators=250, contamination="auto", max_samples=max_samples, max_features=max_features, verbose=1)
clf.fit(X_train)

# predictions
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

print("Inlier Ratio:\t{0:.2f}".format(list(y_pred_test).count(1) / y_pred_test.shape[0]))
print("Outlier Ratio:\t{0:.2f}".format(list(y_pred_outliers).count(-1) / y_pred_outliers.shape[0]))
