#!/usr/bin/python
# -*- coding: utf-8 -*-
import re

# dependence
import MeCab
import json
import ipadic
import six
from unitTest.medinfo_tools import DICT_PATH, STOP_WORDS

# PATH
SPACES_PATTERN = r"[\s\n\r]+"
SPACES_RE = re.compile(SPACES_PATTERN)


def tokenize(text, stemmer):
    text = text.lower()
    vail_tokens = []
    _tokens = SPACES_RE.split(text)
    if stemmer:
        _tokens = [six.ensure_str(stemmer.stem(x)) if len(x) > 3 else x for x in _tokens]
    with open(STOP_WORDS, 'r', encoding='UTF-8') as file:
        stopwords = json.load(file)
    tagger = MeCab.Tagger(ipadic.MECAB_ARGS + f" -O wakati -u {DICT_PATH}")
    for line in _tokens:
        tokens = tagger.parse(line)
        if tokens is None:
            continue
        else:
            tokens = tokens.split()
            tokens = [token for token in tokens if token not in stopwords]
            vail_tokens.extend(tokens)

    return vail_tokens
