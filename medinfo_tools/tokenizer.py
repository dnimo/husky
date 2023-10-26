#!/usr/bin/python
# -*- coding: utf-8 -*-
import re

# dependence
import MeCab
import json
import ipadic
import six

# PATH
DICT_PATH = "data/MANBYO_202106.dic"
STOP_WORDS = "data/ja.json"


def tokenize(text, stemmer):
    if stemmer:
        tokens = []
        tokens = [six.ensure_str(stemmer.stem(x)) if len(x) > 3 else x for x in tokens]
        return tokens
    else:
        valid_tokens = []
        stopwords = json.load(open(STOP_WORDS))
        lines = [line for line in text.splitlines() if line]
        tagger = MeCab.Tagger(ipadic.MECAB_ARGS + f" -O wakati -u {DICT_PATH}")
        for line in lines:
            tokens = tagger.parse(line)
            if tokens is None:
                continue
            else:
                tokens = tokens.split()
                valid_tokens = [token for token in tokens if token not in stopwords]

        return valid_tokens


if __name__ == '__main__':
    out = tokenize("私は京都大学のがくせいです！")
    print(out)
