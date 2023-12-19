import sys

sys.path.append('/home/jovyan/mi-drive/medinfo_lab/Research_Projects/zhang/Husky')

from sklearn.feature_extraction.text import TfidfVectorizer
import MeCab
import ipadic
from tools.tokenizers import myMeCab
from tools.valuations.entropy import calculate_entropy
from joblib import Parallel, delayed


CorpusPath = '/home/jovyan/mi-drive/medinfo_lab/Research_Projects/zhang/Husky/data/del_none_data_for_train_tokenizer.txt'


def load_data(CorpusPath):
    with open(CorpusPath, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data
  
data = load_data(CorpusPath)

def tokenize(data):
    tokenized_data = []
    for line in data:
        _tokens = myMeCab.tokenize(text=line, stemmer=False)
        tokenized_data.extend(_tokens)
    return tokenized_data
  
tfidf_vectorizer = TfidfVectorizer(use_idf=True, tokenizer=tokenize)


def parallelize_vectorization(text):
    return tfidf_vectorizer.fit_transform([text])
  
  
multi_processing = Parallel(n_jobs=3)
tfidf_matrix = multi_processing(delayed(parallelize_vectorization)(text) for text in data)


output = []
sentence_matrix = tfidf_vectorizer.transform(data[:1000]).toarray()
for index, line in enumerate(data[:1000]):
    ens = calculate_entropy(sentence_matrix, batch=True)
    output.append(ens[index], line)
    
    
import matplotlib.pyplot as plt
entropy = []
for item in output:
    entropy.append(item[0])
plt.hist(entropy, bins=80)
plt.xlabel('entropy')
plt.ylabel('count')
plt.title('entropy distribute Plot')
plt.savefig('example_plot.png')
plt.show()