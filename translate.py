import torch
from argparse import ArgumentParser

from corpus.corpus import Corpus
from infer.translate_single import TranslatorSingle
from infer.translate_average import TranslatorAverage
from infer.translate_ensemble import TranslatorEnsemble
from model.model import TransformerMT
from optim.label_smoothing import LabelSmoothing
from utils.bleu import BLEU


argparser = ArgumentParser(description='Transformer')

argparser.add_argument('--prefix', type=str, help='Prefix of all files', default='')

argparser.add_argument('--src_test', type=str, nargs='+', help='Source test file name', required=True)
argparser.add_argument('--tgt_test', type=str, nargs='+', help='Target test file name', required=True)

argparser.add_argument('--src_prefix', type=str, help='Prefix of all source files', default='')
argparser.add_argument('--tgt_prefix', type=str, help='Prefix of all target files', default='')
argparser.add_argument('--src_suffix', type=str, help='Suffix of all source files', default='')
argparser.add_argument('--tgt_suffix', type=str, help='Suffix of all target files', default='')

argparser.add_argument('--src_vocab', type=str, nargs='+', help='Path of source vocabulary', required=True)
argparser.add_argument('--tgt_vocab', type=str, nargs='+', help='Path of target vocabulary', required=True)
argparser.add_argument('--joint_vocab', type=str, nargs='+', help='Path of joint vocabulary', default='')

argparser.add_argument('--model_prefix', type=str, help='Prefix of all model files', default='')
argparser.add_argument('--model_suffix', type=str, help='Suffix of all model files', default='')
argparser.add_argument('--model', type=str, nargs='+', help='Path of storaged model', required=True)
argparser.add_argument('--mode', type=str, choices=['separate', 'average', 'ensemble'],
                       help='Mode of inferring for multi models. If multi paths of models are inputted,'
                            'this denotes the mode of using models. "Separate" means using single model recurrently, '
                            '"average" means averaging all input models, and "ensemble" means ensembling all models '
                            'by averaging the probabilities of softmax logits.',
                       default='separate')
argparser.add_argument('--params', type=str, help='Path of storaged parameters', required=True)

argparser.add_argument('--batch_size', type=int, help='Batch size', default=32)
argparser.add_argument('--beam_size', type=int, nargs='+', help='Beam size of beam search', default=[1])
argparser.add_argument('--decoding_alpha', type=float, nargs='+', help='Length penalty alpha when decoding', default=[1.0])
argparser.add_argument('--infer_max_seq_length', type=int, help='Max length of sequences when translating', default=256)
argparser.add_argument('--infer_max_seq_length_mode', type=str, choices=['relative', 'absolute'],
                       help='Determine "infer_max_seq_length" is used as absolute length or additive relative length. '
                            'For the latter, sequence length will be the sum of source length and "infer_max_seq_length".',
                       default='absolute')

argparser.add_argument('--save_output', help='Whether to save hypothesis to output files', action='store_true')
argparser.add_argument('--output_prefix', type=str, help='Prefix of output files', default='')
argparser.add_argument('--output_suffix', type=str, help='Suffix of output files', default='')

argparser.add_argument('--device', type=int, help='device to use', required=True)

main_args = argparser.parse_args()


def translate():
    if len(main_args.src_test) != len(main_args.tgt_test):
        print('Number of source test files %d does not match with target files %d.'
              % (len(main_args.src_test), len(main_args.tgt_test)))
        return

    src_paths = list(main_args.src_prefix + x + main_args.src_suffix for x in main_args.src_test)
    tgt_paths = list(main_args.tgt_prefix + x + main_args.tgt_suffix for x in main_args.tgt_test)

    args = {'file_prefix': '',
            'num_of_layers': '',
            'num_of_heads': '',
            'src_vocab_size': '',
            'tgt_vocab_size': '',
            'embedding_size': '',
            'applied_bpe': '',
            'bpe_suffix_token': '@@',
            'share_embedding': '',
            'share_projection_and_embedding': '',
            'emb_norm_clip': '',
            'emb_norm_clip_type': '',
            'positional_encoding': '',
            'bpe_src': '',
            'bpe_tgt': '',
            'tgt_character_level': '',
            'src_vocab': '',
            'tgt_vocab': '',
            'joint_vocab': '',
            'feedforward_size': '',
            'layer_norm_pre': '',
            'layer_norm_post': '',
            'layer_norm_encoder_start': '',
            'layer_norm_encoder_end': '',
            'layer_norm_decoder_start': '',
            'layer_norm_decoder_end': '',
            'activate_function_name': '',
            'src_pad_token': '',
            'src_unk_token': '',
            'src_sos_token': '',
            'src_eos_token': '',
            'tgt_pad_token': '',
            'tgt_unk_token': '',
            'tgt_eos_token': '',
            'tgt_sos_token': '',
            'optimizer': '',
            'label_smoothing': ''}

    with open(main_args.params, 'r') as f:
        for _, line in enumerate(f):
            splits = line.split()

            if splits[0] in args.keys():
                if len(splits) == 2:
                    args[splits[0]] = splits[1]
                    if args[splits[0]] == 'True':
                        args[splits[0]] = True
                    elif args[splits[0]] == 'False':
                        args[splits[0]] = False
                elif len(splits) == 1:
                    args[splits[0]] = None

    device = torch.device(main_args.device)

    corpus = Corpus(
        prefix=main_args.prefix,
        corpus_source_train='',
        corpus_source_valid='',
        corpus_source_test=src_paths,
        corpus_target_train='',
        corpus_target_valid='',
        corpus_target_test=tgt_paths,
        bpe_suffix_token=args['bpe_suffix_token'],
        bpe_src=args['bpe_src'],
        bpe_tgt=args['bpe_tgt'],
        share_embedding=args['share_embedding'],
        min_seq_length=1,
        max_seq_length=128,
        batch_size=main_args.batch_size,
        length_merging_mantissa_bits=2,
        src_pad_token=args['src_pad_token'],
        src_unk_token=args['src_unk_token'],
        src_sos_token=args['src_sos_token'],
        src_eos_token=args['src_eos_token'],
        tgt_pad_token=args['tgt_pad_token'],
        tgt_unk_token=args['tgt_unk_token'],
        tgt_sos_token=args['tgt_sos_token'],
        tgt_eos_token=args['tgt_eos_token'],
        logger=None,
        num_of_workers=1,
        num_of_steps=1,
        batch_capacity=1024,
        train_buffer_size=1,
        train_prefetch_size=1,
        device=device)
    corpus.build_vocab(src_vocab_size=0, tgt_vocab_size=0,
                       src_vocab_path=main_args.src_vocab[0],
                       tgt_vocab_path=main_args.tgt_vocab[0],
                       joint_vocab_path=main_args.joint_vocab[0] if args['share_embedding'] else None)
    corpus.test_file_stats()
    corpus.corpus_numerate_test()

    model = TransformerMT(
        src_vocab_size=corpus.src_vocab_size,
        tgt_vocab_size=corpus.tgt_vocab_size,
        joint_vocab_size=corpus.joint_vocab_size,
        share_embedding=args['share_embedding'],
        share_projection_and_embedding=args['share_projection_and_embedding'],
        src_pad_idx=corpus.src_word2idx[corpus.src_pad_token],
        tgt_pad_idx=corpus.tgt_word2idx[corpus.tgt_pad_token],
        tgt_sos_idx=corpus.tgt_word2idx[corpus.tgt_sos_token],
        tgt_eos_idx=corpus.tgt_word2idx[corpus.tgt_eos_token],
        positional_encoding=args['positional_encoding'],
        emb_size=int(args['embedding_size']),
        feed_forward_size=int(args['feedforward_size']),
        num_of_layers=int(args['num_of_layers']),
        num_of_heads=int(args['num_of_heads']),
        train_max_seq_length=128,
        infer_max_seq_length=main_args.infer_max_seq_length,
        infer_max_seq_length_mode=main_args.infer_max_seq_length_mode,
        batch_size=main_args.batch_size,
        embedding_dropout_prob=0.0,
        attention_dropout_prob=0.0,
        feedforward_dropout_prob=0.0,
        residual_dropout_prob=0.0,
        emb_norm_clip=float(args['emb_norm_clip']),
        emb_norm_clip_type=float(args['emb_norm_clip_type']),
        layer_norm_pre=args['layer_norm_pre'],
        layer_norm_post=args['layer_norm_post'],
        layer_norm_encoder_start=args['layer_norm_encoder_start'],
        layer_norm_encoder_end=args['layer_norm_encoder_end'],
        layer_norm_decoder_start=args['layer_norm_decoder_start'],
        layer_norm_decoder_end=args['layer_norm_decoder_end'],
        activate_function_name=args['activate_function_name'],
        prefix=args['file_prefix'],
        pretrained_src_emb='',
        pretrained_tgt_emb='',
        pretrained_src_eos='',
        pretrained_tgt_eos='',
        src_vocab=args['src_vocab'],
        tgt_vocab=args['tgt_vocab'],
        criterion=LabelSmoothing(vocab_size=corpus.tgt_vocab_size,
                                 padding_idx=0,
                                 confidence=1 - float(args['label_smoothing'])),
        update_decay=1
    ).to(device)
    model.eval()

    print(model)
    print('*' * 80)

    bleu = BLEU()

    model_paths = list(main_args.model_prefix + model + main_args.model_suffix for model in main_args.model)
    print('Translate mode: %s' % main_args.mode)

    if main_args.mode == 'separate':
        translator = TranslatorSingle(corpus=corpus,
                                      bleu=bleu,
                                      model=model,
                                      model_paths=model_paths,
                                      src_pad_idx=corpus.src_word2idx[corpus.src_pad_token],
                                      tgt_pad_idx=corpus.src_word2idx[corpus.tgt_pad_token],
                                      tgt_sos_idx=corpus.src_word2idx[corpus.tgt_sos_token],
                                      tgt_eos_idx=corpus.src_word2idx[corpus.tgt_eos_token],
                                      tgt_character_level=args['tgt_character_level'],
                                      beam_size=main_args.beam_size,
                                      decoding_alpha=main_args.decoding_alpha,
                                      save_output=main_args.save_output,
                                      output_prefix=main_args.output_prefix,
                                      output_suffix=main_args.output_suffix)
    elif main_args.mode == 'average':
        translator = TranslatorAverage(corpus=corpus,
                                       bleu=bleu,
                                       model=model,
                                       model_paths=model_paths,
                                       src_pad_idx=corpus.src_word2idx[corpus.src_pad_token],
                                       tgt_pad_idx=corpus.src_word2idx[corpus.tgt_pad_token],
                                       tgt_sos_idx=corpus.src_word2idx[corpus.tgt_sos_token],
                                       tgt_eos_idx=corpus.src_word2idx[corpus.tgt_eos_token],
                                       tgt_character_level=args['tgt_character_level'],
                                       beam_size=main_args.beam_size,
                                       decoding_alpha=main_args.decoding_alpha,
                                       save_output=main_args.save_output,
                                       output_prefix=main_args.output_prefix,
                                       output_suffix=main_args.output_suffix)
    else:
        translator = TranslatorEnsemble(corpus=corpus,
                                        bleu=bleu,
                                        model=model,
                                        model_paths=model_paths,
                                        src_pad_idx=corpus.src_word2idx[corpus.src_pad_token],
                                        tgt_pad_idx=corpus.src_word2idx[corpus.tgt_pad_token],
                                        tgt_sos_idx=corpus.src_word2idx[corpus.tgt_sos_token],
                                        tgt_eos_idx=corpus.src_word2idx[corpus.tgt_eos_token],
                                        tgt_character_level=args['tgt_character_level'],
                                        beam_size=main_args.beam_size,
                                        decoding_alpha=main_args.decoding_alpha,
                                        save_output=main_args.save_output,
                                        output_prefix=main_args.output_prefix,
                                        output_suffix=main_args.output_suffix,
                                        src_vocab_paths=main_args.src_vocab,
                                        tgt_vocab_paths=main_args.tgt_vocab)

    translator.translate()

    return


if __name__ == '__main__':
    translate()
