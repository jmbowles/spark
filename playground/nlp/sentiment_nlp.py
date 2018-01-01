import sys

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf, col
from pyspark.sql.types import *
from sparknlp.annotator import *
from sparknlp.base import DocumentAssembler

''' 

spark-submit --packages JohnSnowLabs:spark-nlp:1.2.3 spark_nlp.py

pyspark --packages JohnSnowLabs:spark-nlp:1.2.3 
execfile('sentiment_nlp.py')

'''

spark = SparkSession\
        .builder\
        .appName("Spark_NLP")\
        .getOrCreate()

data = spark. \
        read. \
        parquet("../datasets/parquet/sentiment.parquet"). \
        limit(1000)
data.cache()
data.count()
data.show()

### Define the dataframe      
document_assembler = DocumentAssembler() \
            .setInputCol("text")
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
    .setPositiveSource("positive") \
    .setNegativeSource("negative") \
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