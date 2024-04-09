from sklearn.feature_extraction.text import TfidfVectorizer
import MeCab
import re
import ipadic

# PATH

file_path = r'C:\Users\KuoChing\workspace\husky\data\sample.txt'

# Load data

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data


data = load_data(file_path)


# Tokenize

def tokenize(data):
    tagger = MeCab.Tagger(ipadic.MECAB_ARGS + " -O wakati")
    tokenized_data = []
    for line in data:
        tokenized_data.extend(tagger.parse(line).split())
    return tokenized_data


tfidf_vectorizer = TfidfVectorizer(use_idf=True, tokenizer=tokenize)

tfidf_matrix = tfidf_vectorizer.fit_transform(data)


sentence_matrix = tfidf_vectorizer.transform(data).toarray()

# Calculate entropy