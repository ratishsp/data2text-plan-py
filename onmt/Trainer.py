from __future__ import division
"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
See train.py for a use case of this library.

Note!!! To make this a general library, we implement *only*
mechanism things here(i.e. what to do), and leave the strategy
things to users(i.e. how to do it). Also see train.py(one of the
users of this library) for the strategy things we do.
"""
import time
import sys
import math
import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.modules


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.ppl(),
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)

    def log_tensorboard(self, prefix, writer, lr, epoch):
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/ppl", self.ppl(), epoch)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), epoch)
        writer.add_scalar(prefix + "/tgtper",  self.n_words / t, epoch)
        writer.add_scalar(prefix + "/lr", lr, epoch)


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train

            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
    """

    def __init__(self, model, model2, train_loss, valid_loss, train_loss2, valid_loss2, optim, optim2,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", grad_accum_count=1, cuda= False):
        # Basic attributes.
        self.model = model
        self.model2 = model2
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.train_loss2 = train_loss2
        self.valid_loss2 = valid_loss2
        self.optim = optim
        self.optim2 = optim2
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.cuda = cuda

        assert(grad_accum_count > 0)
        if grad_accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()
        self.model2.train()

    def train(self, train_iter, epoch, report_func=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        total_stats = Statistics()
        report_stats = Statistics()
        total_stats2 = Statistics()
        report_stats2 = Statistics()
        idx = 0
        true_batchs = []
        accum = 0
        normalization = 0
        try:
            add_on = 0
            if len(train_iter) % self.grad_accum_count > 0:
                add_on += 1
            num_batches = len(train_iter) / self.grad_accum_count + add_on
        except NotImplementedError:
            # Dynamic batching
            num_batches = -1

        for i, batch in enumerate(train_iter):
            cur_dataset = train_iter.get_cur_dataset()
            self.train_loss.cur_dataset = cur_dataset
            self.train_loss2.cur_dataset = cur_dataset

            true_batchs.append(batch)
            accum += 1
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:].data.view(-1) \
                    .ne(self.train_loss.padding_idx).sum()
                normalization += num_tokens
            else:
                normalization += batch.batch_size

            if accum == self.grad_accum_count:
                self._gradient_accumulation(
                        true_batchs, total_stats,
                        report_stats, total_stats2, report_stats2, normalization)

                if report_func is not None:
                    report_stats = report_func(
                            epoch, idx, num_batches,
                            total_stats.start_time, self.optim.lr,
                            report_stats)
                    report_stats2 = report_func(
                            epoch, idx, num_batches,
                            total_stats2.start_time, self.optim2.lr,
                            report_stats2)

                true_batchs = []
                accum = 0
                normalization = 0
                idx += 1

        if len(true_batchs) > 0:
            self._gradient_accumulation(
                    true_batchs, total_stats,
                    report_stats, total_stats2, report_stats2, normalization)
            true_batchs = []

        return total_stats, total_stats2

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()
        self.model2.eval()

        stats = Statistics()
        stats2 = Statistics()

        for batch in valid_iter:
            cur_dataset = valid_iter.get_cur_dataset()
            self.valid_loss.cur_dataset = cur_dataset
            self.valid_loss2.cur_dataset = cur_dataset

            src = onmt.io.make_features(batch, 'src1', self.data_type)
            self.tt = torch.cuda if self.cuda else torch
            src_lengths = self.tt.LongTensor(batch.src1.size()[1]).fill_(batch.src1.size()[0])

            tgt = batch.tgt1_planning.unsqueeze(2)
            # F-prop through the model.
            outputs, attns, _, memory_bank = self.model(src, tgt, src_lengths)
            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(
                    batch, outputs, attns, stage1=True)
            # Update statistics.
            stats.update(batch_stats)

            inp_stage2 = tgt[1:-1]
            index_select = [torch.index_select(a, 0, i).unsqueeze(0) for a, i in
                            zip(torch.transpose(memory_bank, 0, 1), torch.t(torch.squeeze(inp_stage2, 2)))]
            emb = torch.transpose(torch.cat(index_select), 0, 1)
            _, src_lengths = batch.src2
            tgt = onmt.io.make_features(batch, 'tgt2')
            # F-prop through the model.
            outputs, attns, _, _ = self.model2(emb, tgt, src_lengths)
            # Compute loss.
            batch_stats = self.valid_loss2.monolithic_compute_loss(
                batch, outputs, attns, stage1=False)
            # Update statistics.
            stats2.update(batch_stats)

        # Set model back to training mode.
        self.model.train()
        self.model2.train()

        return stats, stats2

    def epoch_step(self, ppl, ppl2, epoch):
        self.optim.update_learning_rate(ppl, epoch)
        self.optim2.update_learning_rate(ppl2, epoch)


    def drop_checkpoint(self, opt, epoch, fields, valid_stats, valid_stats2):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': onmt.io.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
        }
        torch.save(checkpoint,
                   '%s_stage1_acc_%.4f_ppl_%.4f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))

        real_model = (self.model2.module
                      if isinstance(self.model2, nn.DataParallel)
                      else self.model2)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': onmt.io.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim2,
        }
        torch.save(checkpoint,
                   '%s_stage2_acc_%.4f_ppl_%.4f_e%d.pt'
                   % (opt.save_model, valid_stats2.accuracy(),
                      valid_stats2.ppl(), epoch))

    def _gradient_accumulation(self, true_batchs, total_stats,
                               report_stats, total_stats2, report_stats2, normalization):
        if self.grad_accum_count > 1:
            assert False
            self.model.zero_grad()

        for batch in true_batchs:
            #Stage 1
            target_size = batch.tgt1.size(0)

            trunc_size = target_size

            dec_state = None

            src = onmt.io.make_features(batch, 'src1', self.data_type)
            self.tt = torch.cuda if self.cuda else torch
            src_lengths = self.tt.LongTensor(batch.src1.size()[1]).fill_(batch.src1.size()[0])

            for j in range(0, target_size-1, trunc_size):
                #setting to value of tgt_planning
                tgt = batch.tgt1_planning[j: j + trunc_size].unsqueeze(2)

                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model.zero_grad()
                outputs, attns, dec_state, memory_bank = \
                    self.model(src, tgt, src_lengths, dec_state)

                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss.sharded_compute_loss(
                        batch, outputs, attns, j,
                        trunc_size, self.shard_size, normalization, retain_graph=True)

                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

            #Stage 2
            target_size = batch.tgt2.size(0)
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                assert False
                trunc_size = target_size

            dec_state = None
            _, src_lengths = batch.src2
            report_stats2.n_src_words += src_lengths.sum()

            #memory bank is of size src_len*batch_size*dim, inp_stage2 is of size inp_len*batch_size*1
            inp_stage2 = tgt[1:-1]
            index_select = [torch.index_select(a, 0, i).unsqueeze(0) for a, i in
                            zip(torch.transpose(memory_bank, 0, 1), torch.t(torch.squeeze(inp_stage2, 2)))]
            emb = torch.transpose(torch.cat(index_select), 0, 1)
            if self.data_type == 'text':
                tgt_outer = onmt.io.make_features(batch, 'tgt2')
            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model2.zero_grad()
                outputs, attns, dec_state, _ = \
                    self.model2(emb, tgt, src_lengths, dec_state)

                # retain_graph is false for the final truncation
                retain_graph = (j + trunc_size) < (target_size - 1)
                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss2.sharded_compute_loss(
                        batch, outputs, attns, j,
                        trunc_size, self.shard_size, normalization, retain_graph=retain_graph)


                # 4. Update the parameters and statistics.
                if self.grad_accum_count == 1:
                    self.optim2.step()
                total_stats2.update(batch_stats)
                report_stats2.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

            # 4. Update the parameters and statistics.
            self.optim.step()

        if self.grad_accum_count > 1:
            assert False
            self.optim.step()
