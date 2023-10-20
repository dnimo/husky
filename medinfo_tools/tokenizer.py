#!/usr/bin/python
# -*- coding: utf-8 -*-
import re

# dependence
import MeCab
import json
import ipadic

# PATH
DICT_PATH = "data/MANBYO_202106.dic"
STOP_WORDS = "data/ja.json"


class Tokenizer:

    def __int__(self):
        print("loading MeCab tokenizer for Japanese..")

    def tokenized(self, text: str):
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
    tokenizer = Tokenizer()
    out = tokenizer.tokenized("私は京都大学のがくせいです！")
    print(out)
