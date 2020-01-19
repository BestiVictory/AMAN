# -*- coding: utf-8 -*-

from cider_scorer3 import CiderScorer3


class CiderScorerAlias:
    def __init__(self, data_set):
        num_gts = len(data_set[0])
        gts_samples = {}

        # gts[1]: image number, gts[0]: caption.
        for i in range(num_gts):
            gts = data_set[0][i]
            if gts[1] not in gts_samples:
                gts_samples[gts[1]] = [gts[0] + ' <eos>']
            else:
                gts_samples[gts[1]].append(gts[0] + ' <eos>')

        refs_ls = list(gts_samples.values())
        self.scorer = CiderScorer3(refs_ls)
        self.gts = gts_samples

    @staticmethod
    def _to_str(hypo, idx2word):
        s = []
        for k in hypo:
            if k == 0:
                s.append('<eos>')
                break
            s.append(idx2word[1] if k not in idx2word else idx2word[k])
        return ' '.join(s)

    # hypo is a list of numbers.
    def compute_score(self, img_id, hypo, idx2word):
        refs = self.gts[img_id]
        hypo = CiderScorerAlias._to_str(hypo, idx2word)

        return self.scorer.compute_score(hypo, refs)
