"""
This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
#from ctcdecode import CTCBeamDecoder
import Levenshtein as Lev

import onmt
import onmt.inputters as inputters
from onmt.modules.sparse_losses import SparsemaxLoss
from onmt.modules.sparse_activations import LogSparsemax


def build_loss_compute(model, tgt_vocab, opt, train=True, ctc_vocab=None):
    """
    Returns a LossCompute subclass which wraps around an nn.Module subclass
    (such as nn.NLLLoss) which defines the loss criterion. The LossCompute
    object allows this loss to be computed in shards and passes the relevant
    data to a Statistics object which handles training/validation logging.
    Currently, the NMTLossCompute class handles all loss computation except
    for when using a copy mechanism. Despite their name, LossCompute objects
    do not merely compute the loss but also perform the backward pass inside
    their sharded_compute_loss method.
    """
    device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")

    padding_idx = tgt_vocab.stoi[inputters.PAD_WORD]
    if opt.copy_attn:
        criterion = onmt.modules.CopyGeneratorLoss(
            len(tgt_vocab), opt.copy_attn_force,
            unk_index=inputters.UNK, ignore_index=padding_idx
        )
    elif opt.label_smoothing > 0 and train:
        criterion = LabelSmoothingLoss(
            opt.label_smoothing, len(tgt_vocab), ignore_index=padding_idx
        )
    elif isinstance(model.generator[1], LogSparsemax):
        criterion = SparsemaxLoss(ignore_index=padding_idx, reduction='sum')
    else:
        criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')

    # if the loss function operates on vectors of raw logits instead of
    # probabilities, only the first part of the generator needs to be
    # passed to the NMTLossCompute. At the moment, the only supported
    # loss function of this kind is the sparsemax loss.
    use_raw_logits = isinstance(criterion, SparsemaxLoss)
    loss_gen = model.generator[0] if use_raw_logits else model.generator
    if opt.copy_attn:
        compute = onmt.modules.CopyGeneratorLossCompute(
            criterion, loss_gen, tgt_vocab, opt.copy_loss_by_seqlength
        )
    elif opt.ctc_ratio == 0:
        compute = NMTLossCompute(criterion, loss_gen)
    else:
        ctc_gen = model.encoder.ctc_gen
        ctc_crit = nn.CTCLoss(blank=0, reduction='sum', zero_infinity=True)
        if opt.ctc_ratio == 1:
            decoder = None
            compute = CTCLossCompute(ctc_crit, ctc_gen, decoder,
                                     eos_idx=eos_idx)
        else:
            crits = [criterion, ctc_crit]
            gens = [loss_gen, ctc_gen]
            compute = CTCAttLossCompute(crits, gens, opt.ctc_ratio, ctc_vocab=ctc_vocab)
    compute.to(device)

    return compute


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating multiple
    loss computations

    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, criterion, generator):
        super(LossComputeBase, self).__init__()
        self.criterion = criterion
        self.generator = generator

    @property
    def padding_idx(self):
        return self.criterion.ignore_index

    def _make_shard_state(self, batch, output, range_, enc_out, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, enc_out, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, output, enc_out, attns):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.utils.Statistics`: loss statistics
        """
        range_ = (0, batch.tgt.size(0))
        shard_state = self._make_shard_state(batch, output, range_, enc_out, attns)
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    def sharded_compute_loss(self, batch, output, enc_out, attns,
                             cur_trunc, trunc_size, shard_size,
                             normalization, optimizer=None):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.utils.Statistics`: validation loss statistics

        """
        batch_stats = onmt.utils.Statistics()
        range_ = (cur_trunc, cur_trunc + trunc_size)
        shard_state = self._make_shard_state(batch, output, range_, enc_out, attns)
        if optimizer.fp16:
            backward = lambda loss: optimizer.optimizer.backward(loss)
        else:
            backward = lambda loss: loss.backward()

        loss, stats = self._compute_loss(batch, **shard_state)
        loss = loss.div(float(normalization))
        backward(loss)
        batch_stats.update(stats)
        #for shard in shards(shard_state, shard_size):
        #    loss, stats = self._compute_loss(batch, **shard)
        #    loss = loss.div(float(normalization))
        #    backward(loss)
        return batch_stats

    def wer(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1 + s2)
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1]
        w2 = [chr(word2char[w]) for w in s2]

        return Lev.distance(''.join(w1), ''.join(w2))

    def _stats(self, loss, scores, target, ctc_loss=float('-inf'), ctc_target=None):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        #pred = scores.max(-1)[1]
        #non_padding = target.ne(self.padding_idx)
        #pred = pred.t().tolist()
        #target = target.t().tolist()
        #def filtre(p, t):
        #    return list(zip(*[(p, t) for p, t in zip(p, t) if t != self.padding_idx]))

        #num_corrects =  [self.wer(*filtre(p, t)) for p, t in zip(pred, target)]
        #num_correct = sum(num_corrects)
        #num_non_padding = non_padding.sum().item()
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        num_ctc = 0
        if ctc_target is not None:
            num_ctc = ctc_target.ne(self.padding_idx).sum().item()
        return onmt.utils.Statistics(loss.item(), num_non_padding, num_correct,
                                     ctc_loss, num_ctc)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, criterion, generator, normalization="sents"):
        super(NMTLossCompute, self).__init__(criterion, generator)

    def _make_shard_state(self, batch, output, range_, enc_out=None, attns=None):
        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
        }

    def _compute_loss(self, batch, output, target):
        bottled_output = self._bottle(output)

        scores = self.generator(bottled_output)
        gtruth = target.view(-1)

        loss = self.criterion(scores, gtruth)
        stats = self._stats(loss.clone(), scores, gtruth)

        return loss, stats


class CTCLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, criterion, generator, decoder, eos_idx):
        super(CTCLossCompute, self).__init__(criterion, generator)
        self.eos_idx = eos_idx
        self.decoder = decoder

    def _make_shard_state(self, batch, output, range_, enc_out=None, attns=None):
        return {
            "output": output,
            "enc_out": enc_out,
            "target": batch.tgt[range_[0] + 1: range_[1]],
        }

    def _compute_loss(self, batch, output, enc_out, target):
        in_len = enc_out.size(0)
        batch_size = enc_out.size(1)
        input_lengths =  target.new_full((batch_size,), in_len)

        target = target.transpose(0,1)
        valid_indices = target.ne(self.padding_idx) * target.ne(self.eos_idx)
        target_lengths = valid_indices.sum(dim=-1)
        #target = target.masked_select(valid_indices)

        bottled_enc_out = self._bottle(enc_out)

        scores = self.generator(bottled_enc_out)
        scores = scores.view(-1, batch_size, scores.size(-1))

        loss = self.criterion(scores, target, input_lengths, target_lengths)
        stats = self._stats(loss.clone(), scores, target)

        return loss, stats

    def _stats(self, loss, scores, target):

        scores = scores.transpose(0,1).contiguous()
        #beam_results, _, _, out_seq_len = self.decoder.decode(scores)
        #target = [seq.masked_select(mask).tolist() for seq, mask in
        #          zip(target, target.ne(self.padding_idx) * target.ne(self.eos_idx))]
        #results = [result[0][0:l[0]] for result, l in
        #           zip(beam_results.tolist(), out_seq_len.tolist())]
        #dists = [self.wer(s1, s2) for s1, s2 in zip(results, target)]
        #num_non_padding = sum([len(seq) for seq in target])
        #num_correct = num_non_padding - sum(dists)
        #non_padding = target.ne(self.padding_idx)
        #non_eos = target.ne(self.eos_idx)
        #target = target.masked_select(non_padding)
        #num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        #num_non_padding = non_padding.sum().item()
        num_non_padding = 100
        num_correct = 0
        return onmt.utils.Statistics(loss.item(), num_non_padding, num_correct)


class CTCAttLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, criterions, generators, ctc_ratio,
                 normalization="sents", ctc_vocab=None):
        super(CTCAttLossCompute, self).__init__(criterions, generators)
        self.ctc_ratio = ctc_ratio
        self.eos_idx = ctc_vocab.stoi[inputters.EOS_WORD]
        self.bos_idx = ctc_vocab.stoi[inputters.BOS_WORD]
        self.sil_idx = ctc_vocab.stoi['$']

    def _make_shard_state(self, batch, output, range_, enc_out=None, attns=None):
        target = batch.tgt[range_[0] + 1: range_[1]]
        if hasattr(batch, 'tgt_feat_0'):
            ctc_target = batch.tgt_feat_0 # don't apply range on it. that's decoder's.

        return {
            "output": output,
            "enc_out": enc_out,
            "target": target,
            "ctc_target": ctc_target
        }

    @property
    def padding_idx(self):
        return self.criterion[0].ignore_index

    def _compute_loss(self, batch, output, enc_out, target, ctc_target):
        batch_size = enc_out.size(1)
        bottled_output = self._bottle(output)

        scores = self.generator[0](bottled_output)
        gtruth = target.view(-1)

        loss = self.criterion[0](scores, gtruth)
        #scores = scores.view(-1, batch_size, scores.size(-1))

        input_lengths =  torch.full((batch_size,), enc_out.size(0), dtype=torch.int32)

        ctc_target = ctc_target.transpose(0,1)

        valid_indices = ctc_target.ne(self.bos_idx) * \
                        ctc_target.ne(self.eos_idx) * \
                        ctc_target.ne(self.sil_idx) * \
                        ctc_target.ne(self.padding_idx)

        target_lengths = valid_indices.sum(dim=1, dtype=torch.int32).cpu()
        #no_blank = 0 not in ctc_target.contiguous().view(-1)
        #wont_be_nan = (target_lengths + 1 < input_lengths).byte().all() and no_blank
        ctc_loss = 0
        #if wont_be_nan:
        ctc_target = ctc_target.masked_select(valid_indices)
        bottled_enc_out = self._bottle(enc_out)

        ctc_scores = self.generator[1](bottled_enc_out)
        ctc_scores = ctc_scores.view(-1, batch_size, ctc_scores.size(-1))

        ctc_loss = self.criterion[1](ctc_scores, ctc_target,
                                     input_lengths, target_lengths)
        #if not wont_be_nan or ctc_loss == float('inf') or ctc_loss == float('nan'):
        #    print("ctc_loss: {}, skip training".format(ctc_loss))
        #    ctc_loss = 0
        #    ctc_target = None

        stats = self._stats(loss.detach(), scores, gtruth, ctc_loss.detach(), ctc_target)
                #ctc_loss.detach() if wont_be_nan else float('-inf'), ctc_target)
        loss = self.ctc_ratio * ctc_loss + (1 - self.ctc_ratio) * loss

        return loss, stats


def filter_shard_state(state, shard_size=None):
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
