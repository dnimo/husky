#!/usr/bin/python
# -*- coding: utf-8 -*-
# dependence
import MeCab
import json
import ipadic
from tools import DICT_PATH, STOP_WORDS


with open(STOP_WORDS, 'r', encoding='UTF-8') as file:
    stopwords = json.load(file)
tagger = MeCab.Tagger(ipadic.MECAB_ARGS + f" -O wakati {DICT_PATH}")


class MeCabTokenizer:
    @staticmethod
    def tokenize(text: str):
        text = text.lower()
        tokens = tagger.parse(text)
        tokens = tokens.split()
        tokens = [token for token in tokens if token not in stopwords]

        return tokens
