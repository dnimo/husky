import matplotlib.pyplot as plt
import sentencepiece as spm
from nltk import FreqDist
import json


# TODO: Tokens distribution analysis

def tokenize(tokenizer, text):
    result = tokenizer.tokenize(text)
    return {"tokens": result}


class TokensDist():
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def plot(self, data):
        data = data.map(lambda x: tokenize(self.tokenizer, x))
        # tokens =