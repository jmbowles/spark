import sys
import re

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf, col, concat_ws
from pyspark.sql.types import *
from sparknlp.annotator import *
from sparknlp.base import DocumentAssembler

''' 

spark-submit --packages JohnSnowLabs:spark-nlp:1.2.3 spark_nlp.py

pyspark --packages JohnSnowLabs:spark-nlp:1.2.3 
execfile('sentiment_nlp.py')

https://github.com/JohnSnowLabs/spark-nlp/blob/master/python/example/vivekn-sentiment/sentiment.ipynb
spark-submit --packages JohnSnowLabs:spark-nlp:1.2.3 --driver-memory 10g lda_sentiment.py 
pyspark --packages JohnSnowLabs:spark-nlp:1.2.3 --driver-memory 10g
http://nlp.johnsnowlabs.com/components.html

'''

spark = SparkSession\
        .builder\
        .appName("LDA+Sentiment: 20newsgroups")\
        .getOrCreate()


def apply_regex(doc, min_term_length=4):
    """
    Tokenizer to split text based on any whitespace, keeping only terms of at least a certain length which start with an alphabetic character.
    """
    return [x.lower() for x in token_pattern.findall(doc) if (len(x) >= min_term_length and x[0].isalpha())]


token_pattern = re.compile(r"\b\w\w+\b", re.U)

doc_files = spark.sparkContext.wholeTextFiles(path='../datasets/20news/20news-bydate-train/*', minPartitions=8, use_unicode=False).mapValues(lambda doc: apply_regex(doc)).cache()
data = doc_files.toDF(schema=['file', 'doc']).withColumn('doc', concat_ws(' ', 'doc'))


### Define the dataframe      
document_assembler = DocumentAssembler() \
            .setInputCol("doc")
### Transform input to appropriate schema
#assembled = document_assembler.transform(data)

### Sentence detector
sentence_detector = SentenceDetectorModel() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")
#sentence_data = sentence_detector.transform(checked)

tokenizer = RegexTokenizer() \
            .setInputCols(["sentence"]) \
            .setOutputCol("token")
#tokenized = tokenizer.transform(assembled)

### Spell Checker
spell_checker = NorvigSweetingApproach() \
            .setInputCols(["token"]) \
            .setOutputCol("spell")
#checked = spell_checker.fit(tokenized).transform(tokenized)

sentiment_detector = ViveknSentimentApproach() \
    .setInputCols(["spell", "sentence"]) \
    .setOutputCol("sentiment") \
    .setPositiveSource("../datasets/sentiments/pos") \
    .setNegativeSource("../datasets/sentiments/neg") \
    .setPruneCorpus(False)

finisher = Finisher() \
    .setInputCols(["sentiment"]) \
    .setIncludeKeys(True) \
    .setCleanAnnotations(False)

pipeline = Pipeline(stages=[
    document_assembler,
    sentence_detector,
    tokenizer,
    spell_checker,
    sentiment_detector,
    finisher
])

sentiment_data = pipeline.fit(data).transform(data)
sentiment_data.show()