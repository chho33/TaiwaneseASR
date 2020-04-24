""" Translation main class """
from __future__ import unicode_literals, print_function

import torch
import onmt.inputters as inputters


class TranslationBuilder(object):
    """
    Build a word-based translation from the batch output
    of translator and the underlying dictionaries.

    Replacement based on "Addressing the Rare Word
    Problem in Neural Machine Translation" :cite:`Luong2015b`

    Args:
       data (DataSet):
       fields (dict of Fields): data fields
       n_best (int): number of translations produced
       replace_unk (bool): replace unknown words using attention
       has_tgt (bool): will the batch have gold targets
    """

    def __init__(self, data, fields, n_best=1, replace_unk=False,
                 has_tgt=False):
        self.data = data
        self.fields = fields
        self.n_best = n_best
        self.replace_unk = replace_unk
        self.has_tgt = has_tgt

    def _build_target_tokens(self, src, src_vocab, src_raw, pred, attn, tgt_field="tgt"):
        vocab = self.fields[tgt_field].vocab
        tokens = []
        for tok in pred:
            if tok < len(vocab):
                tokens.append(vocab.itos[tok])
            else:
                tokens.append(src_vocab.itos[tok - len(vocab)])
            if tokens[-1] == inputters.EOS_WORD:
                tokens = tokens[:-1]
                break
        if self.replace_unk and (attn is not None) and (src is not None):
            for i in range(len(tokens)):
                if tokens[i] == vocab.itos[inputters.UNK]:
                    _, max_index = attn[i].max(0)
                    tokens[i] = src_raw[max_index.item()]
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert(len(translation_batch["gold_score"]) ==
               len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, attn, gold_score, \
            ctc_preds, ctc_pred_score, \
            ctc_gold_score, indices = list(zip(
            *sorted(zip(translation_batch["predictions"],
                        translation_batch["scores"],
                        translation_batch["attention"],
                        translation_batch["gold_score"],
                        translation_batch["ctc_predictions"],
                        translation_batch["ctc_scores"],
                        translation_batch["ctc_gold_score"],
                        batch.indices.data),
                    key=lambda x: x[-1])))

        # Sorting
        inds, perm = torch.sort(batch.indices.data)
        data_type = self.data.data_type
        if data_type == 'text':
            src = batch.src[0].data.index_select(1, perm)
        else:
            src = None

        ctc_field = "tgt"
        if self.has_tgt:
            tgt = batch.tgt.data.index_select(1, perm)
            if hasattr(batch, 'tgt_feat_0'):
                ctc_field = "tgt_feat_0"
                ctc_tgt = batch.tgt_feat_0.data.index_select(1, perm)
            else:
                ctc_tgt = None
                
        else:
            tgt = None
            ctc_tgt = None

        translations = []
        for b in range(batch_size):
            if data_type == 'text':
                src_vocab = self.data.src_vocabs[inds[b]] \
                    if self.data.src_vocabs else None
                src_raw = self.data.examples[inds[b]].src
            else:
                src_vocab = None
                src_raw = None
            pred_sents = [self._build_target_tokens(
                src[:, b] if src is not None else None,
                src_vocab, src_raw,
                preds[b][n], attn[b][n])
                for n in range(self.n_best)]
            ctc_pred_sents = None
            if len(ctc_preds[b]) > 0:
                ctc_pred_sents = [self._build_target_tokens(
                    src[:, b] if src is not None else None,
                    src_vocab, src_raw,
                    ctc_preds[b][n] if len(ctc_preds[b]) > 0 else None,
                    None, tgt_field=ctc_field)
                    for n in range(self.n_best)]
            gold_sent = None
            ctc_gold_sent = None
            if tgt is not None:
                gold_sent = self._build_target_tokens(
                    src[:, b] if src is not None else None,
                    src_vocab, src_raw,
                    tgt[1:, b] if tgt is not None else None, None)
                if ctc_tgt is not None:
                    ctc_gold_sent = self._build_target_tokens(
                        src[:, b] if src is not None else None,
                        src_vocab, src_raw,
                        ctc_tgt[1:, b] if ctc_tgt is not None else None, None,
                        tgt_field=ctc_field)

            translation = Translation(src[:, b] if src is not None else None,
                                      src_raw, pred_sents,
                                      attn[b], pred_score[b], gold_sent,
                                      gold_score[b], ctc_pred_sents,
                                      ctc_pred_score[b], ctc_gold_sent,
                                      ctc_gold_score[b])
            translations.append(translation)

        return translations


class Translation(object):
    """
    Container for a translated sentence.

    Attributes:
        src (`LongTensor`): src word ids
        src_raw ([str]): raw src words

        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention dist for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """

    def __init__(self, src, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score,
                 ctc_pred_sents, ctc_pred_scores,
                 ctc_tgt_sent, ctc_gold_score):
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score
        self.ctc_pred_sents = ctc_pred_sents
        self.ctc_pred_scores = ctc_pred_scores
        self.ctc_gold_sent = ctc_tgt_sent
        self.ctc_gold_score = ctc_gold_score

    def log(self, sent_number):
        """
        Log translation.
        """

        output = '\nSENT {}: {}\n'.format(sent_number, self.src_raw)

        def _log(pred_sents, pred_scores, gold_sent, gold_score):
            out = ''
            best_pred = pred_sents[0]
            best_score = pred_scores[0]
            pred_sent = ' '.join(best_pred)
            out += 'PRED {}: {}\n'.format(sent_number, pred_sent)
            out += "PRED SCORE: {:.4f}\n".format(best_score)

            if gold_sent is not None:
                tgt_sent = ' '.join(gold_sent)
                out += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
                out += ("GOLD SCORE: {:.4f}\n".format(gold_score))
            if len(pred_sents) > 1:
                out += '\nBEST HYP:\n'
                for score, sent in zip(pred_scores, pred_sents):
                    out += "[{:.4f}] {}\n".format(score, sent)
            return out

        output += _log(self.pred_sents, self.pred_scores,
                self.gold_sent, self.gold_score)
        if self.ctc_pred_sents is not None:
            output += _log(self.ctc_pred_sents, self.ctc_pred_scores,
                    self.ctc_gold_sent, self.ctc_gold_score)

        return output
