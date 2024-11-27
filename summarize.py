import numpy as np
import pandas as pd
import nltk
# Uncomment the following lines if running for the first time
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import math
import os
import pickle

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

# Function to remove stopwords
def remove_stopwords(sentence_tokens, stop_words):
    filtered_sentence = ' '.join([word for word in sentence_tokens if word not in stop_words])
    return filtered_sentence

# Function to generate the summary
def get_summary(scores, sentences, summary_length):
    # Pair each sentence with its score
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    # Select top-ranked sentences
    top_sentences = sorted(ranked_sentences[:summary_length], key=lambda x: sentences.index(x[1]))
    # Concatenate selected sentences to form the summary
    summary = ' '.join([s for score, s in top_sentences])
    return summary

def summarize_text(article_text):
    # Convert text into DataFrame
    df = pd.DataFrame([article_text], columns=['article_text'])

    # Tokenize sentences
    sentences = []
    for s in df['article_text']:
        sentences.extend(sent_tokenize(s))

    if not sentences:
        return "No content to summarize."

    # Determine summary length (35% of total sentences)
    summary_length = max(1, math.ceil(len(sentences) * 0.35))

    # Load or create word embeddings
    embedding_file = os.path.join(THIS_FOLDER, 'dict.pickle')
    if os.path.exists(embedding_file):
        with open(embedding_file, 'rb') as f:
            word_embeddings = pickle.load(f)
    else:
        word_embeddings = {}
        glove_file = os.path.join(THIS_FOLDER, 'model', 'glove.6B.100d.txt')
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                word = parts[0]
                embeddings = np.asarray(parts[1:], dtype='float32')
                word_embeddings[word] = embeddings
        with open(embedding_file, 'wb') as f:
            pickle.dump(word_embeddings, f)

    # Preprocess sentences
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ", regex=True).str.lower()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    clean_sentences = [remove_stopwords(s.split(), stop_words) for s in clean_sentences]

    # Generate sentence vectors
    sentence_vectors = []
    for sentence in clean_sentences:
        words = sentence.split()
        if words:
            vector = sum([word_embeddings.get(w, np.zeros((100,))) for w in words]) / (len(words) + 0.001)
        else:
            vector = np.zeros((100,))
        sentence_vectors.append(vector)

    # Create similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(
                    sentence_vectors[i].reshape(1, 100),
                    sentence_vectors[j].reshape(1, 100)
                )[0, 0]

    # Apply PageRank algorithm
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    # Generate summary
    summary = get_summary(scores, sentences, summary_length)
    return summary