#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import os
# dependence
import MeCab
import json
import ipadic
import six

# PATH
SPACES_PATTERN = r"[\s\n\r]+"
SPACES_RE = re.compile(SPACES_PATTERN)
PATH_DATA = r'C:\Users\KuoChing\workspace\husky\data'

DICT_PATH = os.path.join(PATH_DATA, "MANBYO_202106.dic")
STOP_WORDS = os.path.join(PATH_DATA, "ja.json")


with open(STOP_WORDS, 'r', encoding='UTF-8') as file:
        stopwords = json.load(file)
tagger = MeCab.Tagger(ipadic.MECAB_ARGS + f"-O wakati {DICT_PATH}")


def tokenize(text, stemmer):
    text = text.lower()
    vail_tokens = []
    _tokens = SPACES_RE.split(text)
    if stemmer:
        _tokens = [six.ensure_str(stemmer.stem(x)) if len(x) > 3 else x for x in _tokens]
    for line in _tokens:
        tokens = tagger.parse(line)
        if tokens is None:
            continue
        else:
            tokens = tokens.split()
            tokens = [token for token in tokens if token not in stopwords]
            vail_tokens.extend(tokens)

    return vail_tokens
