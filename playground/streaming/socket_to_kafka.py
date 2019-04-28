"""
 
sudo tcpdump -n -tttt -s 0 -i en1 'tcp port 443 and (((ip[2:2] - ((ip[0]&0xf)<<2)) - ((tcp[12]&0xf0)>>2)) != 0) and dst not 192.168.254.11'
sudo tcpdump -n -tttt -s 0 -i en1 'tcp and (((ip[2:2] - ((ip[0]&0xf)<<2)) - ((tcp[12]&0xf0)>>2)) != 0) and dst not 192.168.254.11' | nc -lk 9999

Druid Process:

bin/supervise -c quickstart/tutorial/conf/tutorial-cluster.conf

Druid Console:

http://localhost:8081/#/datasources/tcp_connections

Druid Kafka Ingestion
curl -XPOST -H'Content-Type: application/json' -d @quickstart/tutorial/tcpdump-kafka-supervisor.json http://localhost:8090/druid/indexer/v1/supervisor

Kafka:

./bin/kafka-server-start.sh config/server.properties
./bin/kafka-topics.sh --list --zookeeper localhost:2181
./bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic tcp_connections
./bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic tcp_connections

Spark Submittal:

spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.0

Turnilo Process:

turnilo --druid http://localhost:8082

Turnilo UI:
http://localhost:9090

Example Data Published to Kafka:

{"event_ts":"2019-04-28T12:54:19.170-04:00","source_ip":"192.168.254.11","source_port":53247,"dest_ip":"208.80.154.224","dest_port":443,"bytes_sent":45}

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

'''
parsed.writeStream\
    .outputMode("append")\
    .format("console")\
    .option("truncate", False)\
    .option("numRows", 250)\
    .option("checkpointLocation", "../datasets/checkpoints/socket_to_kafka_console")\
    .start()
'''

parsed.selectExpr("to_json(struct(*)) AS value")\
    .writeStream\
    .format("kafka")\
    .outputMode("append")\
    .trigger(processingTime="15 seconds")\
    .option("kafka.bootstrap.servers", "localhost:9092")\
    .option("topic", "tcp_connections")\
    .option("checkpointLocation", "../datasets/checkpoints/socket_to_kafka_publish")\
    .start()

spark.streams.awaitAnyTermination()