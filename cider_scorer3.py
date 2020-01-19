# -*- coding: utf-8 -*-
# Tsung-Yi Lin <tl483@cornell.edu>
# Ramakrishna Vedantam <vrama91@vt.edu>

from collections import defaultdict
import numpy as np
import math


def precook(s, n=4):
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4):
    return [precook(ref, n) for ref in refs]


def cook_test(test, n=4):
    return precook(test, n)


class CiderScorer3(object):
    def __init__(self, refs_ls=None, n=4, sigma=6.0):
        self.n = n
        self.sigma = sigma
        self.document_frequency = defaultdict(float)
        self.crefs = []

        self._init_corpus(refs_ls)
        # compute idf
        self._compute_doc_freq()

        # compute log reference length
        self.ref_len = np.log(float(len(self.crefs)))

    def _init_corpus(self, refs_ls):
        for refs in refs_ls:
            self.crefs.append(cook_refs(refs))

    def _compute_doc_freq(self):
        for refs in self.crefs:
            # refs, k ref captions of one image
            for ngram in set(
                [ngram for ref in refs for (ngram, count) in ref.items()]
            ):
                self.document_frequency[ngram] += 1

    def compute_cider(self, hypo, refs):
        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            delta = float(length_hyp - length_ref)
            # measure consine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for (ngram, count) in vec_hyp[n].items():
                    # vrama91 : added clipping
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]
                                 ) * vec_ref[n][ngram]

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n] * norm_ref[n])

                assert (not math.isnan(val[n]))
                # vrama91: added a length based gaussian penalty
                val[n] *= np.e**(-(delta**2) / (2 * self.sigma**2))
            return val

        def counts2vec(cnts):
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram, term_freq) in cnts.items():
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, self.document_frequency[ngram]))

                # ngram index
                n = len(ngram) - 1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq) * (self.ref_len - df)
                # compute norm for the vector.  the norm will be used for computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        # compute vector for test captions
        vec, norm, length = counts2vec(hypo)
        score = np.array([0.0 for _ in range(self.n)])
        for ref in refs:
            vec_ref, norm_ref, length_ref = counts2vec(ref)
            score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)

        # change by vrama91 - mean of ngram scores, instead of sum
        score_avg = np.mean(score)
        # divide by number of references
        score_avg /= len(refs)
        # multiply score by 10
        score_avg *= 10.0
        return score_avg

    def compute_score(self, hypo, refs):
        hypo = cook_test(hypo)
        refs = cook_refs(refs)
        # compute cider score
        return self.compute_cider(hypo, refs)
