"""

Execute from within pyspark CLI:

result = spark.sql('select * from parquet.`{}`'.format('datasets/parquet/tcpdump.parquet'))
result.createOrReplaceGlobalTempView('tcpdump')
spark.sql('select * from global_temp.tcpdump').show()

For spark-sql, need to be in the same directory where this is launched from. Cant seem to launch correctly from any other directory
even though same path to spark.sql.warehouse.dir is specified via:

spark-sql --conf spark.sql.warehouse.dir=../datasets/spark-warehouse

"""
from __future__ import print_function

import sys
from pyspark.sql import SparkSession


if __name__ == "__main__":
    

    spark = SparkSession\
        .builder\
        .appName("Show Parquet Data")\
        .enableHiveSupport() \
        .getOrCreate()

result = spark.sql('select * from parquet.`{}`'.format('../datasets/parquet/sentiment.parquet'))
result.write.saveAsTable('tcpdump', mode='overwrite')
result.show(truncate=False)

#result.createOrReplaceGlobalTempView('tcpdump')
#spark.sql('select * from global_temp.tcpdump').show()
    



