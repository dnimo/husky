"""
Author = KuoCh'ing Chang(kuochingcha@gamil.com)
Data = 2023/20/27
New feature -> merge blue to medinfo_tools package make it eazsy to use.
"""
from __future__ import print_function, division

import numpy as np


class BLEU:
    def __init__(self, n_gram=1):
        self.n_gram = n_gram

    def evaluate(self, candidates, references):
        """compute bleu
        @param candidates [[[token1, token2, ...]]] output by model
        @param references [[[token1, token2, ...]]] reference text
        @param bleu result of computing format [candidate_1_result, candidate_2_result]
        """

        BP = 1
        bleu = np.zeros(len(candidates))
        for k, candidate in enumerate(candidates):
            r, c = 0, 0
            count = np.zeros(self.n_gram)
            count_clip = np.zeros(self.n_gram)
            # count_index = np.zeros(self.n_gram)
            p = np.zeros(self.n_gram)

            for j, candidate_sent in enumerate(candidate):
                # for each sentence
                for i in range(self.n_gram):
                    count_, n_grams = self.extractNgram(candidate_sent, i + 1)
                    count[i] += count_
                    reference_sents = []
                    reference_sents = [reference[j] for reference in references]
                    count_clip_, count_index_ = self.countClip(reference_sents, i + 1, n_grams)
                    count_clip[i] += count_clip_
                    c += len(candidate_sent)
                    r += len(reference_sents[count_index_])
                p = count_clip / count
            rc = r / c
            if rc >= 1:
                BP = np.exp(1 - rc)
            else:
                rc = 1

            p[p == 0] = 1e-100
            p = np.log(p)
            bleu[k] = BP * np.exp(np.average(p))
        return bleu

    def extractNgram(self, candidate, n):
        """extractNgram
        @param candidate: [str] output by model
        @param n int n-gram
        @return count int n-gram-numbers
        @return n_grams set() n-grams
        """

        count = 0
        n_grams = set()
        if len(candidate) - n + 1 > 0:
            count += len(candidate) - n + 1
        for i in range(len(candidate) - n + 1):
            n_gram = ' '.join(candidate[i:i + n])
            n_grams.add(n_gram)
        return count, n_grams

    def countClip(self, references, n, n_gram):
        """ count n_grams number in references
        @param references [[str]] reference text
        @param n int n-gram numbers
        @param n_gram set n-grams

        @return:
        @count times
        @index index of max appear
        """

        max_count = 0
        index = 0
        for j, reference in enumerate(references):
            count = 0
            for i in range(len(reference) - n + 1):
                if (' '.join(reference[i:i + n]) in n_gram):
                    count += 1
            if max_count < count:
                max_count = count
                index = j
        return max_count, index
