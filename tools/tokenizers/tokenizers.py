import abc
from nltk.stem import porter
from . import myMeCab


class tokenizers(abc.ABC):

    @abc.abstractmethod
    def tokenize(self, text):
        raise NotImplementedError("Tokenizer must override tokenize method")


class MeCabTokenizer(tokenizers):
    def __init__(self, use_stemmer=False):
        self._stemmer = porter.PorterStemmer() if use_stemmer else None

    def tokenize(self, text):
        return myMeCab.tokenize(text, self._stemmer)


class DefaultTokenizer(tokenizers):
    def __init__(self, use_stemmer=False):
        self._stemmer = porter.PorterStemmer() if use_stemmer else None

    def tokenize(self, text):
        return myMeCab.tokenize(text, self._stemmer)


if __name__ == '__main__':
    print("hello tokenizer")
