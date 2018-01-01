from __future__ import division

from Bio import Entrez, Medline
import numpy as np
import pandas as pd
import pyspark
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.clustering import LDA
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType, IntegerType

''' 
https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/6665730192733135/2604545365219192/7713747021737742/latest.html
pip install biopython

spark-submit --packages JohnSnowLabs:spark-nlp:1.2.3 lda.py

pyspark --packages JohnSnowLabs:spark-nlp:1.2.3 
execfile('lda.py')

https://spark.apache.org/docs/2.2.0/ml-pipeline.html
https://gist.github.com/Bergvca/a59b127afe46c1c1c479
https://community.hortonworks.com/questions/130866/rowwise-manipulation-of-a-dataframe-in-pyspark.html
https://stackoverflow.com/questions/42284681/udf-to-map-words-to-term-index-in-spark

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

topic_terms = [
    ['type', '1', 'diabetes'], 
    ['creutzfeldt', 'jakob', 'disease'], 
    ['post', 'traumatic', 'stress', 'disorder'],
    ['heart', 'disease'],
    ['AIDS'],
    ['breast', 'cancer']]


def getDocuments(topic_terms):
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

def indices_to_terms(vocab):
    def indices_to_terms(indices):
        return [vocab[int(index)] for index in indices]
    return udf(indices_to_terms, ArrayType(StringType()))

def topic_index():
    def argmax(topic_probs):
        return int(np.argmax(topic_probs))
    return udf(argmax, IntegerType())

#texts = getDocuments(topics)
texts = spark.read.parquet('../datasets/parquet/pubmed.parquet')
texts = texts.filter('text != ""').persist()
texts.count()

tokenizer = Tokenizer(inputCol="text", outputCol="words")

sw_remover = StopWordsRemover()\
    .setInputCol('words')\
    .setOutputCol('cleanwords')\
    .setStopWords(StopWordsRemover.loadDefaultStopWords('english'))

count_v = CountVectorizer()\
    .setInputCol('cleanwords')\
    .setOutputCol('features')

words = tokenizer.transform(texts)
clean_words = sw_remover.transform(words)
features_model = count_v.fit(clean_words)
features = features_model.transform(clean_words)

lda = LDA(k=6, maxIter=10).setSeed(100)
model = lda.fit(features)

topics = model.describeTopics(maxTermsPerTopic=10)
topics = topics.withColumn('topic_words', indices_to_terms(features_model.vocabulary)('termIndices'))
print('The topics described by their top-weighted terms:')
topics.show()

ll = model.logLikelihood(features)
lp = model.logPerplexity(features)
print 'The lower bound on the log likelihood of the entire corpus: {}'.format(str(ll))
print 'The upper bound on perplexity: {}'.format(str(lp))

# Shows the result
model_results = model.transform(features)
results = model_results.withColumn('topic_index', topic_index()('topicDistribution'))
results.show()
