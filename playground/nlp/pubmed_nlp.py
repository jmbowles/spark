from __future__ import division, print_function

from Bio import Entrez, Medline
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import classification_report

import pyspark
from pyspark.ml import Pipeline, feature as spark_ft
from pyspark.sql.functions import udf, col
from pyspark.sql.types import *

from sparknlp.annotator import *
from sparknlp.base import DocumentAssembler

''' 
https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/6665730192733135/2604545365219192/7713747021737742/latest.html
pip install biopython

spark-submit --packages JohnSnowLabs:spark-nlp:1.2.3 pubmed_nlp.py

pyspark --packages JohnSnowLabs:spark-nlp:1.2.3 
execfile('pubmed_nlp.py')

'''

def query(terms, num_docs=1000):
    search_term = '+'.join(terms)
    print('Searching PubMed abstracts for documents containing term: ',search_term)
    handle = Entrez.esearch(db="pubmed", term=search_term, retmax=num_docs)
    record = Entrez.read(handle)
    handle.close()
    idlist = record["IdList"]
    
    handle = Entrez.efetch(db="pubmed", id=idlist, rettype="medline",retmode="text")
    records = Medline.parse(handle)
    data = []
    for record in records:
        data.append((record.get("TI", "?"),record.get("AU", "?"),record.get("SO", "?"),record.get("AB","?")))

    df = pd.DataFrame(data=data, columns=['title','authors','source','text'])
    df.head(10)

    df.replace(r'^\?$', np.nan, regex=True, inplace=True)
    df['authors'] = df['authors'].apply(lambda x: x if isinstance(x, list) else [])
    df.fillna('', inplace=True)
    df['topic'] = search_term
    
    return spark.createDataFrame(df)

topics = [
    ['type', '1', 'diabetes'], 
    ['creutzfeldt', 'jakob', 'disease'], 
    ['post', 'traumatic', 'stress', 'disorder'],
    ['heart', 'disease'],
    ['AIDS'],
    ['breast', 'cancer']]


def getDocuments(topics):
    np.random.seed(123)
    texts = None
    for terms in topics:
        num_docs = np.random.randint(200, 1000)
        print('terms', terms, 'num_docs', num_docs)
        if texts is None:
            texts = query(terms, num_docs)
        else:
            texts = texts.union(query(terms, num_docs))
    return texts

#texts = getDocuments(topics)
texts = spark.read.parquet('../datasets/parquet/pubmed.parquet')
texts = texts.filter('text != ""').persist()
texts.count()
train, test = texts.randomSplit(weights=[0.8, 0.2], seed=123)
vocab_size = 500

document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
sentence_detector = SentenceDetectorModel().setInputCols(["document"]).setOutputCol("sentence")
tokenizer = RegexTokenizer().setInputCols(["sentence"]).setOutputCol("token")
finisher = Finisher().setInputCols(['token']).setOutputCols(['tokens']).setOutputAsArray(True).setIncludeKeys(True)

sw_remover = spark_ft.StopWordsRemover()\
    .setInputCol('tokens')\
    .setOutputCol('cleantokens')\
    .setStopWords(spark_ft.StopWordsRemover.loadDefaultStopWords('english'))

hashingtf = spark_ft.HashingTF()\
    .setInputCol('cleantokens')\
    .setOutputCol('tf')\
    .setNumFeatures(vocab_size)

idf = spark_ft.IDF()\
    .setInputCol('tf')\
    .setOutputCol('tfidf')

label_indexer = spark_ft.StringIndexer(inputCol='topic', outputCol='label')
pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, finisher, sw_remover, hashingtf, idf, label_indexer])
pipeline_model = pipeline.fit(train)
tx_train = pipeline_model.transform(train)
train_df = tx_train.select('title', 'label', 'tfidf').toPandas()
train_df.head()