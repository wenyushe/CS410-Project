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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Function to filter out stopwords
def filter_stopwords(token_list, stopword_set):
    filtered_text = ' '.join([word for word in token_list if word not in stopword_set])
    return filtered_text

# Function to create a summary from scores and sentences
def construct_summary(pagerank_scores, sentence_list, summary_size):
    # Pair each sentence with its score
    ranked_sentences = sorted(((pagerank_scores[i], s) for i, s in enumerate(sentence_list)), reverse=True)
    # Select the top-ranked sentences
    top_sentences = sorted(ranked_sentences[:summary_size], key=lambda x: sentence_list.index(x[1]))
    # Combine the top sentences into the summary
    summary_text = ' '.join([s for score, s in top_sentences])
    return summary_text

def generate_summary(input_text):
    # Convert the input into a DataFrame
    text_dataframe = pd.DataFrame([input_text], columns=['full_text'])

    # Tokenize the text into sentences
    sentence_list = []
    for text in text_dataframe['full_text']:
        sentence_list.extend(sent_tokenize(text))

    # Calculate the desired summary length
    desired_summary_length = max(1, math.ceil(np.log(len(sentence_list))))

    word_vectors = {}
    glove_path = os.path.join(BASE_DIR, 'model', 'glove.6B.100d.txt')
    with open(glove_path, 'r', encoding='utf-8') as glove_file:
        for line in glove_file:
            elements = line.split()
            word = elements[0]
            vector = np.asarray(elements[1:], dtype='float32')
            word_vectors[word] = vector

    # Preprocess the sentences
    cleaned_sentences = pd.Series(sentence_list).str.replace("[^a-zA-Z]", " ", regex=True).str.lower()

    # Remove stopwords from sentences
    stopword_set = set(stopwords.words('english'))
    cleaned_sentences = [filter_stopwords(sentence.split(), stopword_set) for sentence in cleaned_sentences]

    # Create sentence embeddings
    sentence_embeddings = []
    for sentence in cleaned_sentences:
        word_list = sentence.split()
        if word_list:
            embedding = sum([word_vectors.get(word, np.zeros((100,))) for word in word_list]) / (len(word_list) + 0.001)
        else:
            embedding = np.zeros((100,))
        sentence_embeddings.append(embedding)

    # Build similarity matrix
    similarity_matrix = np.zeros([len(sentence_list), len(sentence_list)])
    for i in range(len(sentence_list)):
        for j in range(len(sentence_list)):
            if i != j:
                similarity_matrix[i][j] = cosine_similarity(
                    sentence_embeddings[i].reshape(1, 100),
                    sentence_embeddings[j].reshape(1, 100)
                )[0, 0]

    # Apply PageRank to the similarity graph
    graph = nx.from_numpy_array(similarity_matrix)
    pagerank_scores = nx.pagerank(graph)

    # Generate the final summary
    summary_output = construct_summary(pagerank_scores, sentence_list, desired_summary_length)
    return summary_output
