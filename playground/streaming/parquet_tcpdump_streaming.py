#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
 Counts words in UTF8 encoded, '\n' delimited text received from the network.
 Usage: structured_network_wordcount.py <hostname> <port>
   <hostname> and <port> describe the TCP server that Structured Streaming
   would connect to receive data.


 https://thorstenball.com/blog/2013/08/11/named-pipes/
 http://technicallysane.blogspot.com/p/using-tcpdump-with-netcat.html
 https://www.g-loaded.eu/2006/11/06/netcat-a-couple-of-useful-examples/
 
 df.select(regexp_extract(df.s, 'IP\s(.+?)\s>\s(.+?)\sFlags',2).alias('s')).collect()

 sudo tcpdump -tttt -s 0 -i en0 tcp and not dst host localhost | nc -lk 9999
 spark-submit parsed_tcpdump_streaming.py localhost 9999

"""
from __future__ import print_function

import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split
from pyspark.sql.functions import explode, split, regexp_extract, unix_timestamp, avg, count, minute, hour, window

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: tcpdump_socket.py <hostname> <port>", file=sys.stderr)
        exit(-1)

    host = sys.argv[1]
    port = int(sys.argv[2])

    spark = SparkSession\
        .builder\
        .appName("TCPDump Streaming")\
        .getOrCreate()

    # Create DataFrame representing the stream of input lines from connection to host:port
    raw = spark\
        .readStream\
        .format('socket')\
        .option('host', host)\
        .option('port', port)\
        .load()

    # Split the lines into words
    parsed = raw.select(unix_timestamp(regexp_extract('value', r'^(\d+-\d+-\d+\s\d+:\d+:\d+).+\sIP', 1)).cast('timestamp').alias('unix_timestamp'),
                        regexp_extract('value', r'^(\d+-\d+-\d+\s\d+:\d+:\d+).+\sIP', 1).alias('timestamp'),
                        regexp_extract('value', r'IP\s(.+?)\s>\s.+\sFlags', 1).alias('src'),
                        regexp_extract('value', r'IP\s.+\s>\s(.+?)\sFlags', 1).alias('dst'))

    outbound = parsed.withWatermark('unix_timestamp','1 minute').groupBy(window('unix_timestamp', '15 seconds', '10 seconds'), 'src','dst').agg(count("*").alias('total_count'))

    
    # query = outbound\
    #     .writeStream\
    #     .outputMode('complete')\
    #     .format('console')\
    #     .option('truncate', False)\
    #     .option("numRows", 250)\
    #     .start()

    query2 = outbound\
    .writeStream\
    .format('parquet')\
    .option('checkpointLocation', '../datasets/checkpoints/tcpdump')\
    .trigger(processingTime='15 seconds')\
    .partitionBy('src')\
    .start('../datasets/parquet/tcpdump.parquet')

    #query.awaitTermination()
    query2.awaitTermination()