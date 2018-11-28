"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.Models
import onmt.modules
from onmt.Models import NMTModel, MeanEncoder, RNNEncoder, PointerRNNDecoder,\
                        StdRNNDecoder, InputFeedRNNDecoder
from onmt.modules import Embeddings, ImageEncoder, CopyGenerator, \
                         TransformerEncoder, TransformerDecoder, \
                         CNNEncoder, CNNDecoder, AudioEncoder
from onmt.Utils import use_gpu


def make_embeddings(opt, word_dict, feature_dicts, for_encoder=True):
    """
    Make an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): make Embeddings for encoder or decoder?
    """
    if for_encoder:
        embedding_dim = opt.src_word_vec_size
    else:
        embedding_dim = opt.tgt_word_vec_size

    word_padding_idx = word_dict.stoi[onmt.io.PAD_WORD]
    num_word_embeddings = len(word_dict)

    feats_padding_idx = [feat_dict.stoi[onmt.io.PAD_WORD]
                         for feat_dict in feature_dicts]
    num_feat_embeddings = [len(feat_dict) for feat_dict in
                           feature_dicts]

    return Embeddings(word_vec_size=embedding_dim,
                      position_encoding=opt.position_encoding,
                      feat_merge=opt.feat_merge,
                      feat_vec_exponent=opt.feat_vec_exponent,
                      feat_vec_size=opt.feat_vec_size,
                      dropout=opt.dropout,
                      word_padding_idx=word_padding_idx,
                      feat_padding_idx=feats_padding_idx,
                      word_vocab_size=num_word_embeddings,
                      feat_vocab_sizes=num_feat_embeddings)


def make_encoder(opt, embeddings, stage1=True):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
        stage1: stage1 encoder
    """

    if stage1:
        return MeanEncoder(opt.enc_layers1, embeddings, opt.src_word_vec_size, opt.attn_hidden, opt.dropout)
    else:
        # "rnn" or "brnn"
        return RNNEncoder(opt.rnn_type, opt.brnn2, opt.enc_layers2,
                          opt.rnn_size, opt.dropout, embeddings,
                          opt.bridge)


def make_decoder(opt, embeddings, stage1):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
        stage1: stage1 decoder
    """
    if stage1:
        return PointerRNNDecoder(opt.rnn_type, opt.brnn,
                             opt.dec_layers1, opt.rnn_size,
                             opt.global_attention,
                             opt.coverage_attn,
                             opt.context_gate,
                             False,
                             opt.dropout,
                             embeddings,
                             False,
                             opt.decoder_type1)
    else:
        return InputFeedRNNDecoder(opt.rnn_type, opt.brnn2,
                                   opt.dec_layers2, opt.rnn_size,
                                   opt.global_attention,
                                   opt.coverage_attn,
                                   opt.context_gate,
                                   True,
                                   opt.dropout,
                                   embeddings,
                                   opt.reuse_copy_attn)

def load_test_model(opt, dummy_opt, stage1=False):
    opt_model = opt.model if stage1 else opt.model2
    checkpoint = torch.load(opt_model,
                            map_location=lambda storage, loc: storage)
    fields = onmt.io.load_fields_from_vocab(
        checkpoint['vocab'], data_type=opt.data_type)

    model_opt = checkpoint['opt']
    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]

    model = make_base_model(model_opt, fields,
                            use_gpu(opt), checkpoint, stage1)
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


def make_base_model(model_opt, fields, gpu, checkpoint=None, stage1=True):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """
    if stage1:
        src = "src1"
        tgt = "tgt1"
    else:
        src = "src2"
        tgt = "tgt2"
    assert model_opt.model_type in ["text", "img", "audio"], \
        ("Unsupported model type %s" % (model_opt.model_type))

    # Make encoder.
    if model_opt.model_type == "text":
        src_dict = fields[src].vocab
        feature_dicts = onmt.io.collect_feature_vocabs(fields, src)
        src_embeddings = make_embeddings(model_opt, src_dict,
                                         feature_dicts)
        encoder = make_encoder(model_opt, src_embeddings, stage1)
    elif model_opt.model_type == "img":
        encoder = ImageEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout)
    elif model_opt.model_type == "audio":
        encoder = AudioEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,
                               model_opt.sample_rate,
                               model_opt.window_size)

    # Make decoder.
    tgt_dict = fields[tgt].vocab
    feature_dicts = onmt.io.collect_feature_vocabs(fields, tgt)
    tgt_embeddings = make_embeddings(model_opt, tgt_dict,
                                     feature_dicts, for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        if src_dict != tgt_dict:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')

        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

    decoder = make_decoder(model_opt, tgt_embeddings, stage1)

    # Make NMTModel(= encoder + decoder).
    model = NMTModel(encoder, decoder)
    model.model_type = model_opt.model_type

    # Make Generator.
    if stage1:
        generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, len(fields["tgt1"].vocab)),
            nn.LogSoftmax())
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        generator = CopyGenerator(model_opt.rnn_size,
                                  fields["tgt2"].vocab)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

    # Add generator to model (this registers it as parameter of model).
    model.generator = generator

    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model
