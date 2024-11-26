import numpy as np
import pandas as pd
import nltk
# nltk.download('punkt_tab') # one time execution
# nltk.download('stopwords') # one time execution
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import math
import re
import sys
import os
import pickle

THIS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)))

# function to remove stopwords
def remove_stopwords(sen, stop_words):
  sen_new = ' '.join([i for i in sen if i not in stop_words])
  return sen_new

# returns the top SN sentences in order
def get_summary(scores, sentences, SN):
  scored_sentences = sorted(((score, sentence) for score, sentence in zip(scores, sentences)), reverse=True)
  top_sentences = set([sentence for score, sentence in scored_sentences[:SN]])
  summarized_filename = ' - '.join([filename, "summary"])
  with open(summarized_filename, 'w') as f:
    summary_text = '\n'.join([sentence for sentence in sentences if sentence in top_sentences])
    f.write(summary_text)

def summarize(article_file):
  with open(article_file, 'r') as f:
    df = pd.DataFrame([f.read().replace('\n', ' ')], columns=['article_text'])

  sentences = []
  for s in df['article_text']:
    sentences.append(sent_tokenize(s))

  sentences = [y for x in sentences for y in x] # flatten list

  SN = math.ceil(len(sentences) * .35)

  # remove punctuations, numbers and special characters
  clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

  # make alphabets lowercase
  clean_sentences = [s.lower() for s in clean_sentences]

  # get stopwords
  stop_words = stopwords.words('english')

  # remove stopwords from the sentences
  clean_sentences = [remove_stopwords(r.split(), stop_words) for r in clean_sentences]
  for sentence in clean_sentences:
    print([sentence])



# check for proper usage
if len(sys.argv) != 2:
  print("Usage: summarize.py article_to_summarize")
else:
  filename = sys.argv[1]
  if not os.path.isfile(filename):
    print("Article to summarize does not exist")
  else:
    summarize(filename)