#!/usr/bin/env python

from __future__ import division, unicode_literals
import os
import argparse
import math
import codecs
import torch

from itertools import count

import onmt.io
import onmt.translate
import onmt
import onmt.ModelConstructor
import onmt.modules
import opts

parser = argparse.ArgumentParser(
    description='translate.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opts.add_md_help_argument(parser)
opts.translate_opts(parser)

opt = parser.parse_args()


def _report_score(name, score_total, words_total):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / words_total,
        name, math.exp(-score_total / words_total)))


def _report_bleu():
    import subprocess
    print()
    res = subprocess.check_output(
        "perl tools/multi-bleu.perl %s < %s" % (opt.tgt, opt.output),
        shell=True).decode("utf-8")
    print(">> " + res.strip())


def _report_rouge():
    import subprocess
    res = subprocess.check_output(
        "python tools/test_rouge.py -r %s -c %s" % (opt.tgt, opt.output),
        shell=True).decode("utf-8")
    print(res.strip())


def main():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    # Load the model.
    fields, model, model_opt = \
        onmt.ModelConstructor.load_test_model(opt, dummy_opt.__dict__, stage1=True)

    model2 = None
    if opt.model2 is not None:
        fields2, model2, model_opt2 = \
            onmt.ModelConstructor.load_test_model(opt, dummy_opt.__dict__, stage1=False)

    # File to write sentences to.
    out_file = codecs.open(opt.output, 'w', 'utf-8')

    # Test data
    data = onmt.io.build_dataset(fields, opt.data_type,
                                 opt.src1, opt.tgt1,
                                 opt.src2, opt.tgt2,
                                 src_dir=opt.src_dir,
                                 sample_rate=opt.sample_rate,
                                 window_size=opt.window_size,
                                 window_stride=opt.window_stride,
                                 window=opt.window,
                                 use_filter_pred=False)

    def sort_minibatch_key(ex):
        """ Sort using length of source sentences and length of target sentence """
        #Needed for packed sequence
        if hasattr(ex, "tgt1"):
            return len(ex.src1), len(ex.tgt1)
        return len(ex.src1)

    # Sort batch by decreasing lengths of sentence required by pytorch.
    # sort=False means "Use dataset's sortkey instead of iterator's".
    data_iter = onmt.io.OrderedIterator(
        dataset=data, device=opt.gpu,
        batch_size=opt.batch_size, train=False, sort=False,
        sort_key=sort_minibatch_key,
        sort_within_batch=True, shuffle=False)

    # Translator
    scorer = onmt.translate.GNMTGlobalScorer(opt.alpha,
                                             opt.beta,
                                             opt.coverage_penalty,
                                             opt.length_penalty)
    tgt_plan_map = None

    if opt.src2 is None:
        tgt_plan_map = {}
        for j, entry in enumerate(fields["tgt1"].vocab.itos):
            if j<4:
                tgt_plan_map[j] = j
            else:
                tgt_plan_map[j] = int(entry)
    translator = onmt.translate.Translator(
        model, model2, fields,
        beam_size=opt.beam_size,
        n_best=opt.n_best,
        global_scorer=scorer,
        max_length=opt.max_length,
        copy_attn=model_opt.copy_attn and tgt_plan_map is None,
        cuda=opt.cuda,
        beam_trace=opt.dump_beam != "",
        min_length=opt.min_length,
        stepwise_penalty=opt.stepwise_penalty)
    builder = onmt.translate.TranslationBuilder(
        data, translator.fields,
        opt.n_best, opt.replace_unk, has_tgt=False)

    # Statistics
    counter = count(1)
    pred_score_total, pred_words_total = 0, 0
    gold_score_total, gold_words_total = 0, 0
    stage1 = opt.stage1
    for batch in data_iter:
        batch_data = translator.translate_batch(batch, data, stage1)
        translations = builder.from_batch(batch_data, stage1)

        for trans in translations:
            pred_score_total += trans.pred_scores[0]
            pred_words_total += len(trans.pred_sents[0])
            if opt.tgt2:
                gold_score_total += trans.gold_score
                gold_words_total += len(trans.gold_sent)

            if stage1:
                n_best_preds = [" ".join([str(entry) for entry in pred])
                                for pred in trans.pred_sents[:opt.n_best]]
            else:
                n_best_preds = [" ".join(pred)
                                for pred in trans.pred_sents[:opt.n_best]]
            out_file.write('\n'.join(n_best_preds))
            out_file.write('\n')
            out_file.flush()

            if opt.verbose:
                sent_number = next(counter)
                output = trans.log(sent_number)
                os.write(1, output.encode('utf-8'))

    _report_score('PRED', pred_score_total, pred_words_total)
    if opt.tgt2:
        _report_score('GOLD', gold_score_total, gold_words_total)
        if opt.report_bleu:
            _report_bleu()
        if opt.report_rouge:
            _report_rouge()

    if opt.dump_beam:
        import json
        json.dump(translator.beam_accum,
                  codecs.open(opt.dump_beam, 'w', 'utf-8'))


if __name__ == "__main__":
    main()
