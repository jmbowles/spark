from __future__ import division

import re
import numpy as np
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.clustering import LDA
from pyspark.sql.functions import udf, concat_ws
from pyspark.sql.types import ArrayType, StringType, IntegerType

''' 


'''

spark = SparkSession\
        .builder\
        .appName("LDA: 20newsgroups")\
        .getOrCreate()

def indices_to_terms(vocab):
    def indices_to_terms(indices):
        return [vocab[int(index)] for index in indices]
    return udf(indices_to_terms, ArrayType(StringType()))

def topic_index():
    def argmax(topic_probs):
        return int(np.argmax(topic_probs))
    return udf(argmax, IntegerType())

def apply_regex(doc, min_term_length=4):
    """
    Tokenizer to split text based on any whitespace, keeping only terms of at least a certain length which start with an alphabetic character.
    """
    return [x.lower() for x in token_pattern.findall(doc) if (len(x) >= min_term_length and x[0].isalpha())]


token_pattern = re.compile(r"\b\w\w+\b", re.U)

doc_files = spark.sparkContext.wholeTextFiles(path='../datasets/20news/20news-bydate-train/*', minPartitions=8, use_unicode=False).mapValues(lambda doc: apply_regex(doc)).cache()
docs = doc_files.toDF(schema=['file', 'doc']).withColumn('doc', concat_ws(' ', 'doc'))

tokenizer = Tokenizer(inputCol='doc', outputCol='words')

sw_remover = StopWordsRemover()\
    .setInputCol('words')\
    .setOutputCol('cleanwords')\
    .setStopWords(StopWordsRemover.loadDefaultStopWords('english'))

count_v = CountVectorizer()\
    .setInputCol('cleanwords')\
    .setOutputCol('features')

words = tokenizer.transform(docs)
clean_words = sw_remover.transform(words)
features_model = count_v.fit(clean_words)
features = features_model.transform(clean_words)

lda = LDA(k=20, maxIter=30, optimizer='online').setSeed(100)
model = lda.fit(features)

topics = model.describeTopics(maxTermsPerTopic=10)
topics = topics.withColumn('topic_words', indices_to_terms(features_model.vocabulary)('termIndices'))
print('The topics described by their top-weighted terms:')
topics.show()

# Shows the result
model_results = model.transform(features)
results = model_results.withColumn('topic_index', topic_index()('topicDistribution'))
results.show()

joined = results.join(topics, results.topic_index == topics.topic)
joined = joined.select('file', 'topic', 'topic_words')
joined.show()

