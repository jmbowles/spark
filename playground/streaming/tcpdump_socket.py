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
 spark-submit tcpdump_socket.py localhost 9999

 +-----------------------------------------+-----+
|lines                                    |count|
+-----------------------------------------+-----+
|17.125.249.11.https:                     |14   |
|ord38s04-in-f6.1e100.net.https:          |2    |
|edge-star-shv-02-ort2.facebook.com.https:|3124 |
|17.248.142.87.https:                     |51   |
|ord36s04-in-f110.1e100.net.https:        |134  |
|ord37s18-in-f14.1e100.net.https:         |33   |
|17.188.166.21.5223:                      |14   |
|151.101.65.69.https:                     |3    |
|ord36s04-in-f6.1e100.net.https:          |2    |
|atl14s64-in-f4.1e100.net.https:          |2    |
|ord30s21-in-f78.1e100.net.https:         |30   |
|17.120.252.13.https:                     |43   |
|ord38s08-in-f14.1e100.net.https:         |93   |
|104.16.111.18.https:                     |2    |
|ord37s07-in-f46.1e100.net.https:         |51   |
|ord36s02-in-f14.1e100.net.https:         |88   |
|vl-in-f138.1e100.net.https:              |42   |
|api-chi.smoot.apple.com.https:           |106  |
|17.139.246.5.https:                      |14   |
|151.101.1.69.https:                      |9    |
+-----------------------------------------+-----+
only showing top 20 rows

2017-12-20 06:57:21.931867 IP 192.168.254.10.49807 > edge-star-shv-02-ort2.facebook.com.https: Flags [R], seq 949779844, win 0, length 0
2017-12-20 06:57:21.931868 IP 192.168.254.10.49807 > edge-star-shv-02-ort2.facebook.com.https: Flags [R], seq 949779844, win 0, length 0
2017-12-20 06:57:21.931960 IP 192.168.254.10.49807 > edge-star-shv-02-ort2.facebook.com.https: Flags [R], seq 949779844, win 0, length 0
 

 To run this on your local machine, you need to first run a Netcat server
    $ nc -lk 9999`
 and then run the example
    $ bin/spark-submit examples/src/main/python/sql/streaming/tcpdump_socket.py localhost 9999

"""
from __future__ import print_function

import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split
from pyspark.sql.functions import regexp_extract

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: tcpdump_socket.py <hostname> <port>", file=sys.stderr)
        exit(-1)

    host = sys.argv[1]
    port = int(sys.argv[2])

    spark = SparkSession\
        .builder\
        .appName("TCP Dump Analysis")\
        .getOrCreate()

    # Create DataFrame representing the stream of input lines from connection to host:port
    lines = spark\
        .readStream\
        .format('socket')\
        .option('host', host)\
        .option('port', port)\
        .load()

    # Split the lines into words
    words = lines.select(
        # explode turns each item in an array into a separate row
        regexp_extract(lines.value, 'IP\s(.+?)\s>\s(.+?)\sFlags', 2).alias('lines')
    )


    # Generate running word count
    wordCounts = words.groupBy('lines').count()

    #Start running the query that prints the running counts to the console
    query = wordCounts\
        .writeStream\
        .outputMode('complete')\
        .format('console')\
        .option('truncate', 'false')\
        .start()

    query.awaitTermination()