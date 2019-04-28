"""
 
sudo tcpdump -n -tttt -s 0 -i en1 'tcp port 443 and (((ip[2:2] - ((ip[0]&0xf)<<2)) - ((tcp[12]&0xf0)>>2)) != 0) and dst not 192.168.254.11'
sudo tcpdump -n -tttt -s 0 -i en1 'tcp and (((ip[2:2] - ((ip[0]&0xf)<<2)) - ((tcp[12]&0xf0)>>2)) != 0) and dst not 192.168.254.11' | nc -lk 9999

Druid:

bin/supervise -c quickstart/tutorial/conf/tutorial-cluster.conf

Kafka:

./bin/kafka-server-start.sh config/server.properties
./bin/kafka-topics.sh --list --zookeeper localhost:2181
./bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic tcp_connections

"""
from __future__ import print_function

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

host = "localhost"
port = 9999

spark = SparkSession.builder.appName("TCP Dump Streaming Connections").getOrCreate()

# Create DataFrame representing the stream of input lines from connection to host:port
raw = spark\
    .readStream\
    .format("socket")\
    .option("host", host)\
    .option("port", port)\
    .load()


parsed = raw.withColumn("event_ts", F.regexp_extract("value", r"^(\d+-\d+-\d+\s\d+:\d+:\d+.\d{6})", 1).cast("timestamp"))
parsed = parsed.withColumn("raw_source", F.regexp_extract("value", r"IP\s(.+?)\s>\s.+\sFlags", 1))
parsed = parsed.withColumn("raw_dest", F.regexp_extract("value", r"IP\s.+\s>\s(.+?)\sFlags", 1))
parsed = parsed.withColumn("source_ip", F.substring_index(parsed.raw_source, ".", 4))
parsed = parsed.withColumn("source_port", F.substring_index(parsed.raw_source, ".", -1).cast("integer"))
parsed = parsed.withColumn("dest_ip", F.substring_index(parsed.raw_dest, ".", 4))
parsed = parsed.withColumn("dest_port", F.substring_index(F.substring_index(parsed.raw_dest, ".", -1), ":", 1).cast("integer"))
parsed = parsed.withColumn("bytes_sent", F.regexp_extract("value", r"length\s(\d+)", 1).cast("long"))

cols = ["event_ts", "source_ip", "source_port", "dest_ip", "dest_port", "bytes_sent"]
parsed = parsed.select(*cols).where(F.expr("trim(source_ip) <> ''"))

parsed.writeStream\
    .outputMode("append")\
    .format("console")\
    .option("truncate", False)\
    .option("numRows", 250)\
    .start()

parsed.selectExpr("CAST(id AS STRING) AS key", "to_json(struct(*)) AS value")\
    .writeStream\
    .format("kafka")\
    .outputMode("append")\
    .option("kafka.bootstrap.servers", "192.168.254.11:9092")\
    .option("topic", "tcp_connections")\
    .start()

spark.streams.awaitAnyTermination()