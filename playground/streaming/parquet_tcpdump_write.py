from __future__ import print_function

import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, regexp_extract, unix_timestamp, avg, count, minute, hour, window

if __name__ == "__main__":
    

    spark = SparkSession\
        .builder\
        .appName("Test Parsing TcpDump")\
        .getOrCreate()

    # Populate the following file with some actual tcpdump data.
    raw = spark.read.format('text').load('../datasets/stage/tcpdump_data.txt')
    raw.show(truncate=False)
    

    parsed = raw.select(unix_timestamp(regexp_extract('value', r'^(\d+-\d+-\d+\s\d+:\d+:\d+).+\sIP', 1)).cast('timestamp').alias('unix_timestamp'),
    					regexp_extract('value', r'^(\d+-\d+-\d+\s\d+:\d+:\d+).+\sIP', 1).alias('timestamp'),
    					regexp_extract('value', r'IP\s(.+?)\s>\s.+\sFlags', 1).alias('src'),
    					regexp_extract('value', r'IP\s.+\s>\s(.+?)\sFlags', 1).alias('dst'))
    parsed.show(truncate=False)

    #outbound = parsed.groupBy(hour('timestamp'), 'src','dst').agg(count("*"))
    outbound = parsed.groupBy(window('unix_timestamp', '5 seconds'), 'src','dst').agg(count('*').alias('total_count')).orderBy('window')
    #outbound = outbound.withColumn('unix_timestamp', outbound.unix_timestamp.cast('date'))
    outbound.show(truncate=False)

outbound.write.format('parquet').save('../datasets/stage/parquet/tcpdump.parquet')