# coding=utf-8

import abc
from nltk.stem import porter
import tokenizer


class Tokenizer(abc.ABC):

    @abc.abstractmethod
    def tokenize(self, text):
        raise NotImplementedError("Tokenizer must override tokenize method")


class DefaultTokenizer(Tokenizer):

    def __init__(self, use_stemmer=False):
        self._stemmer = porter.PorterStemmer() if use_stemmer else None

    def tokenize(self, text):
        return tokenizer.tokenize(text, self._stemmer)


if __name__ == '__main__':
    out = DefaultTokenizer(use_stemmer=True).tokenize(text="私は京都大学のがくせいです！")
    print(out)
