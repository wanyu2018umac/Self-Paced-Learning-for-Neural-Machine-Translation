import torch
from torch.multiprocessing import set_sharing_strategy, set_start_method
from torch.optim import Adam, SGD

import resource
from argparse import ArgumentParser

from corpus.corpus import Corpus
from model.model import TransformerMT
from utils.stats import Stats
from utils.bleu import BLEU
from optim.label_smoothing import LabelSmoothing
from train.train_single import TrainerSingleDevice
from train.train_multi import TrainerMultiDevice


argparser = ArgumentParser(description='TransformerMT')

group = argparser.add_argument_group(title='Training information', description='Print infos')
group.add_argument('--annotate', type=str, help='This annotate will be printed after every eval step', default='')

group = argparser.add_argument_group(title='Corpus paths', description='Paths of input corpora.')
group.add_argument('--file_prefix', type=str, help='Prefix of the same directory of all files', required=True)
group.add_argument('--src_train', type=str, help='Source train file name', required=True)
group.add_argument('--src_valid', type=str, help='Source valid file name', required=True)
group.add_argument('--tgt_train', type=str, help='Target train file name', required=True)
group.add_argument('--tgt_valid', type=str,
                   help='Target valid file name. If you have multiple reference files, '
                        'you shuold concatenate all of them, keeping the order corresponding to source file.',
                   required=True)
group.add_argument('--save_vocab', help='Save vocabularies or not', action='store_true')
group.add_argument('--save_corpus', help='Save enumerated train corpus or not', action='store_true')
group.add_argument('--num_of_workers', type=int, help='Number of processes when numerating corpus', default=8)

group = argparser.add_argument_group(title='Vocabulary settings',
                                     description='Preprocessed vocabularies, shared vocab or not, etc.')
group.add_argument('--share_embedding', help='Share source and target embeddings', action='store_true')
group.add_argument('--share_projection_and_embedding',
                   help='Share parameters of projection layer and target embeddings. If optin \"share embedding\" '
                        'is activated, then share parameters of projection layer and shared embeddings.',
                   action='store_true')
group.add_argument('--src_vocab', type=str, help='Source vocab path', default='')
group.add_argument('--tgt_vocab', type=str, help='Target vocab path', default='')
group.add_argument('--joint_vocab', type=str, help='Joint vocab path', default='')

group = argparser.add_argument_group(title='Corpus settings',
                                     description='Preprocessed enumerate corpora.')
group.add_argument('--src_enumerate_corpus', type=str, help='Enumerated source corpus', default='')
group.add_argument('--tgt_enumerate_corpus', type=str, help='Enumerated target corpus', default='')

group = argparser.add_argument_group(title='Retraining', description='For further training.')
group.add_argument('--retrain_model', type=str, help='Path of file which storages model', default='')
group.add_argument('--retrain_params', type=str, help='Path of file which storages parameters', default='')
group.add_argument('--processed_steps', type=int, help='How many steps model is already trained', default=0)

group = argparser.add_argument_group(title='Pretrained parameters', description='Load pretrained embeddings.')
group.add_argument('--pretrained_src_emb', type=str, help='Pretrained source embeddings', default='')
group.add_argument('--pretrained_tgt_emb', type=str, help='Pretrained target embeddings', default='')
group.add_argument('--pretrained_src_eos', type=str,
                   help='Pretrained source embeddings, which token denotes end of sentence for source', default='')
group.add_argument('--pretrained_tgt_eos', type=str,
                   help='Pretrained target embeddings, which token denotes start of sentence for target', default='')

group = argparser.add_argument_group(title='BPE', description='Byte-pair encoding settings. '
                                                              'Strongly suggest to ues fastBPE.')
group.add_argument('--bpe_suffix_token', type=str, help='The token to definite end of subwords except last one',
                   default='@@')
group.add_argument('--bpe_src', help='Whether to use bpe in src side', action='store_true')
group.add_argument('--bpe_tgt', help='Whether to use bpe in tgt side', action='store_true')
group.add_argument('--tgt_character_level',
                   help='Whether to evaluate performance at character-level. '
                        'This is especially essential for some Asian languages such as Chinese.',
                   action='store_true')

group = argparser.add_argument_group(title='Positional encoding', description='Set positional encoding format.')
group.add_argument('--positional_encoding', type=str, help='Positional encoding added to src/tgt embeddings',
                   choices=['none', 'static', 'learnable'], default='static')

group = argparser.add_argument_group(title='Special tokens', description='Set special tokens for training.')
group.add_argument('--src_pad_token', type=str, help='Special token denoting padded tokens for source',
                   default='<PAD>', required=True)
group.add_argument('--src_unk_token', type=str, help='Special token denoting words out of vocabulary for source',
                   default='<UNK>', required=True)
group.add_argument('--src_sos_token', type=str, help='Special token denoting start of sentence for source',
                   default='<SOS>')
group.add_argument('--src_eos_token', type=str, help='Special token denoting end of sentence for source',
                   default='<EOS>', required=True)
group.add_argument('--tgt_pad_token', type=str, help='Special token denoting padded tokens for target',
                   default='<PAD>', required=True)
group.add_argument('--tgt_unk_token', type=str, help='Special token denoting words out of vocabulary for target',
                   default='<UNK>', required=True)
group.add_argument('--tgt_sos_token', type=str, help='Special token denoting start of sentence for target',
                   default='<SOS>', required=True)
group.add_argument('--tgt_eos_token', type=str, help='Special token denoting end of sentence for target',
                   default='<EOS>', required=True)

group = argparser.add_argument_group(title='Model architecture hyper-parameters',
                                     description='Hyper parameters for model architecture.')
group.add_argument('--num_of_layers', type=int, help='Number of layers both in encoder and decoder', default=6)
group.add_argument('--num_of_heads', type=int, help='Number of heads in multihead attention layer', default=8)
group.add_argument('--feedforward_size', type=int, help='Position-wised feed forward size', default=2048)
group.add_argument('--embedding_size', type=int, help='Embedding size', default=512)
group.add_argument('--layer_norm_pre', type=str, help='Layer normalization at the start of each block',
                   choices=['none', 'static', 'learnable'], default='learnable')
group.add_argument('--layer_norm_post', type=str, help='Layer normalization at the end of each block',
                   choices=['none', 'static', 'learnable'], default='none')
group.add_argument('--layer_norm_encoder_start', type=str, help='Layer normalization at the start of encoder',
                   choices=['none', 'static', 'learnable'], default='none')
group.add_argument('--layer_norm_encoder_end', type=str, help='Layer normalization at the start of encoder',
                   choices=['none', 'static', 'learnable'], default='learnable')
group.add_argument('--layer_norm_decoder_start', type=str, help='Layer normalization at the start of decoder',
                   choices=['none', 'static', 'learnable'], default='none')
group.add_argument('--layer_norm_decoder_end', type=str, help='Layer normalization at the start of decoder',
                   choices=['none', 'static', 'learnable'], default='learnable')
group.add_argument('--activate_function_name', type=str, help='Activate function using in feed-forward networks',
                   choices=['relu', 'gelu', 'softplus'], default='relu')

group = argparser.add_argument_group(title='Training hyper-parameters',
                                     description='Hyper parameters for training process.')
group.add_argument('--train_min_seq_length', type=int, help='Min length of sequence when training', default=8)
group.add_argument('--train_max_seq_length', type=int, help='Max length of sequence when training', default=256)
group.add_argument('--infer_max_seq_length', type=int, help='Max length of sequence when translating', default=64)
group.add_argument('--infer_max_seq_length_mode', type=str, choices=['relative', 'absolute'],
                   help='Determine "infer_max_seq_length" is used as absolute length or additive relative length. '
                        'For the latter, sequence length will be the sum of source length and "infer_max_seq_length".',
                   default='absolute')
group.add_argument('--length_merging_mantissa_bits', type=int,
                   help='Mantissa bits for merging examples with close lengths into the same bucket.', default=2)
group.add_argument('--src_vocab_size', type=int, help='Source vocab size', default=50000)
group.add_argument('--tgt_vocab_size', type=int, help='Target vocab size', default=50000)
group.add_argument('--batch_size', type=int, help='Batch size', default=32)
group.add_argument('--batch_capacity', type=int, help='Maximun number of src + tgt tokens in one batch', default=50000)
group.add_argument('--emb_norm_clip', type=float, help='Clip embeddings with maximun norm of each,'
                                                       'a float not greater than 0.0 means disabled', default=0.0)
group.add_argument('--emb_norm_clip_type', type=float, help='P of p-norm when clipping embeddings', default=2.0)
group.add_argument('--grad_norm_clip', type=float, help='Clip gradients with maximun norm of each set of gradient,'
                                                        'a float not greater than 0.0 means disabled', default=0.0)
group.add_argument('--grad_norm_clip_type', type=float, help='P of p-norm when clipping gradients', default=2.0)

group = argparser.add_argument_group(title='Model regularization', description='Model regularization settings.')
group.add_argument('--embedding_keep_prob', type=float,
                   help='Keep probability of source/target word embeddings', default=0.9)
group.add_argument('--attention_keep_prob', type=float,
                   help='Keep probability of self-attention alignment scores', default=0.9)
group.add_argument('--feedforward_keep_prob', type=float,
                   help='Keep probability of feedforward networks', default=0.9)
group.add_argument('--residual_keep_prob', type=float,
                   help='Keep probability of residual connections', default=0.9)
group.add_argument('--merge_dropout_mask', help='Whether to use the same dropout mask for same keep probs above',
                   action='store_true')
group.add_argument('--label_smoothing', type=float,
                   help='Smooth the labels of ground truth when computing loss', default=0.1)
group.add_argument('--l1_scale', type=float, help='Add scaled l1 norm to loss', default=0.0)
group.add_argument('--l2_scale', type=float, help='Add scaled l2 norm to loss', default=1e-6)

group = argparser.add_argument_group(title='Optimizer', description='Optimizer settings.')
group.add_argument('--optimizer', type=str, help='Type of optimizer',
                   choices=['Adam', 'AMSGrad', 'SGD'], default='Adam')
group.add_argument('--learning_rate_schedule', type=str,
                   help='The learning rate schedule of optimizer. '
                        'This is a lambda expression, where parameter in this denotes the present training step, e.g., '
                        'original learning rate schedule in Transformer is: '
                        '"lambda x: 512 ** -0.5 * (x * 4000 ** -1.5 if x < 4000 else x ** -0.5)".',
                   default='lambda x: 2.0 * 512 ** -0.5 * (x * 12000 ** -1.5 if x < 12000 else x ** -0.5)')
group.add_argument('--adam_betas', type=float, nargs=2, help='Betas for Adam', default=[0.9, 0.98])

group = argparser.add_argument_group(title='Training checkpoints',
                                     description='Settings for buffering, reporting, evaluating and training.')
group.add_argument('--buffer_every_steps', type=int, help='Buffer every n batches when training', default=10)
group.add_argument('--prefetch_every_steps', type=int, help='Prefetch every n batches when training', default=10)
group.add_argument('--report_every_steps', type=int, help='Print log after every n steps', default=10)
group.add_argument('--save_every_steps', type=int, help='Save model after every n steps', default=1000)
group.add_argument('--eval_every_steps', type=int, help='Evaluate model after every n steps', default=1000)
group.add_argument('--max_save_models', type=int,
                   help='The maximun number of saved models in this training process for saving hardware space, '
                        'value below than 1 means disabled', default=10)
group.add_argument('--eval_type', type=str, choices=['acc', 'xent', 'bleu'], help='Evaluation type of model training',
                   default='bleu')
group.add_argument('--num_of_steps', type=int, help='Number of training steps', default=100000)
group.add_argument('--update_decay', type=int,
                   help='Update model after every n batches, this will merge generated gradients'
                        'of n batches and update once', default=1)
group.add_argument('--beam_size', type=int, help='Beam size for beam search', default=1)
group.add_argument('--decoding_alpha', type=float, help='Length penalty alpha when translating', default=1.0)

group = argparser.add_argument_group(title='Training devices', description='Device settings.')
group.add_argument('--device', type=int, nargs='+',
                   help='Indexes for gpu devices, the first index will be the main gpu for validating, '
                        'and a negative value denotes cpu device', default=[0])
group.add_argument('--training_batch_chunks_ratio', nargs='+', type=float,
                   help='The ratio of each chunk when splitting one batch into multiple devices.',
                   default=[])

group = argparser.add_argument_group(title='Reproducible', description='Reproducing settings.')
group.add_argument('--random_seed', type=int, help='random seed', default=0)


group = argparser.add_argument_group(title='Self-paced learning parameters', description='Parameters for self-paced learning.')
group.add_argument('--sample_times', type=int, help='Monte Carlo Sampling times', default=5)
group.add_argument('--exponential_value', type=int, help='Exponential value for confidence computation', default=2)
group.add_argument('--normalized_cl_factors', action='store_true', 
                    help='Whether to use softmax to normalize sampled confidence probabilities')

args = argparser.parse_args()

if __name__ == '__main__':
    print('*' * 80)
    print('Set multiprocessing start method to spawn ... ', end='')
    set_start_method('spawn', force=True)
    print('done.')

    print('Set ulimit to maximum ... ', end='')
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))
    print('done.')

    print('Set sharing strategy to file_system ... ', end='')
    set_sharing_strategy('file_system')
    print('done.')

    print('*' * 80)

    if len(args.retrain_model) != 0:
        stored_args = {}
        banned_args = {'retrain_model',
                       'retrain_params',
                       'processed_steps',
                       'num_of_steps',
                       'device',
                       'update_decay',
                       'src_vocab',
                       'tgt_vocab',
                       'joint_vocab'}
        nested_args = {'learning_rate_betas': float}

        with open(args.retrain_params, 'r') as f:
            for _, line in enumerate(f):
                splits = line.split()

                if len(splits) > 1:
                    stored_args[splits[0]] = ' '.join(splits[1:])
                elif len(splits) == 1:
                    stored_args[splits[0]] = None
                else:
                    print('Unrecognized parameter: %s, ignored.' % line)

        for (key, value) in stored_args.items():
            if key in args.__dict__.keys() and key not in banned_args:
                if key in nested_args:
                    recur_type = nested_args[key]
                    args.__dict__[key] = list(recur_type(x.strip()) for x in value[1:-1].split(','))
                else:
                    type_of_value = type(args.__dict__[key])
                    if type_of_value == bool:
                        args.__dict__[key] = True if value == 'True' else False
                    elif type_of_value == int:
                        args.__dict__[key] = int(value)
                    elif type_of_value == float:
                        args.__dict__[key] = float(value)
                    elif type_of_value == str:
                        if value == 'None':
                            args.__dict__[key] = None
                        elif not value:
                            args.__dict__[key] = ''
                        else:
                            args.__dict__[key] = value

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    bleu = BLEU()
    stats = Stats(args.num_of_steps, args.processed_steps)

    device_idx = args.device

    if any(x < 0 for x in device_idx):
        raise ValueError('Not supported for cpu mode.')
    if any(x < 0 for x in device_idx) and not all(x < 0 for x in device_idx):
        raise ValueError('Not supported for cpu + gpu mode.')

    if len(device_idx) == 1:
        if device_idx[0] < 0:
            corpus_device = torch.device('cpu')
            model_device = torch.device('cpu')
        else:
            corpus_device = torch.device(device_idx[0])
            model_device = torch.device(device_idx[0])
    else:
        corpus_device = torch.device('cpu')
        model_device = torch.device(device_idx[0])

    with open(stats.fold_name + '/params.txt', mode='w', encoding='utf-8') as f:
        for key in args.__dict__.keys():
            f.write(str(key) + ' ' + str(args.__dict__[key]) + '\n')

    corpus = Corpus(
        prefix=args.file_prefix,
        corpus_source_train=args.src_train,
        corpus_source_valid=args.src_valid,
        corpus_source_test='',
        corpus_target_train=args.tgt_train,
        corpus_target_valid=args.tgt_valid,
        corpus_target_test='',
        src_pad_token=args.src_pad_token,
        src_unk_token=args.src_unk_token,
        src_sos_token=args.src_sos_token,
        src_eos_token=args.src_eos_token,
        tgt_pad_token=args.tgt_pad_token,
        tgt_unk_token=args.tgt_unk_token,
        tgt_sos_token=args.tgt_sos_token,
        tgt_eos_token=args.tgt_eos_token,
        share_embedding=args.share_embedding,
        bpe_suffix_token=args.bpe_suffix_token,
        bpe_src=args.bpe_src,
        bpe_tgt=args.bpe_tgt,
        min_seq_length=args.train_min_seq_length,
        max_seq_length=args.train_max_seq_length,
        length_merging_mantissa_bits=args.length_merging_mantissa_bits,
        batch_size=args.batch_size,
        logger=stats,
        num_of_workers=args.num_of_workers,
        num_of_steps=args.num_of_steps,
        batch_capacity=args.batch_capacity,
        train_buffer_size=args.buffer_every_steps,
        train_prefetch_size=args.prefetch_every_steps,
        device=corpus_device)

    corpus.train_file_stats()
    corpus.valid_file_stats()
    corpus.build_vocab(args.src_vocab_size, args.tgt_vocab_size, args.src_vocab, args.tgt_vocab, args.joint_vocab)
    corpus.corpus_numerate_train(args.src_enumerate_corpus, args.tgt_enumerate_corpus)
    corpus.corpus_train_lengths_sorting()
    corpus.corpus_numerate_valid()
    corpus.valid_batch_making(args.batch_size)

    if args.save_vocab:
        print('Save src vocab to: ' + stats.fold_name + '/src_vocab.pt')
        print('Save tgt vocab to: ' + stats.fold_name + '/tgt_vocab.pt')
        stats.log_to_file('Save src vocab to: ' + stats.fold_name + '/src_vocab.pt')
        stats.log_to_file('Save tgt vocab to: ' + stats.fold_name + '/tgt_vocab.pt')
        if corpus.share_embedding:
            print('Save joint vocab to: ' + stats.fold_name + '/joint_vocab.pt')
            stats.log_to_file('Save joint vocab to: ' + stats.fold_name + '/joint_vocab.pt')

        torch.save(corpus.src_word2idx, stats.fold_name + '/src_vocab.pt')
        torch.save(corpus.tgt_word2idx, stats.fold_name + '/tgt_vocab.pt')
        if corpus.share_embedding:
            torch.save(corpus.joint_word2idx, stats.fold_name + '/joint_vocab.pt')

    if args.save_corpus:
        print('src train enumerate: ' + '\t' + stats.fold_name + '/src_train_enumerate.pt')
        print('tgt train enumerate: ' + '\t' + stats.fold_name + '/tgt_train_enumerate.pt')
        torch.save(corpus.corpus_source_train_numerate, stats.fold_name + '/src_train_enumerate.pt')
        torch.save(corpus.corpus_target_train_numerate, stats.fold_name + '/tgt_train_enumerate.pt')

    stats.log_to_file('*' * 80)
    print('*' * 80)

    criterion = LabelSmoothing(
        vocab_size=corpus.joint_vocab_size if corpus.share_embedding else corpus.tgt_vocab_size,
        padding_idx=corpus.tgt_word2idx[corpus.tgt_pad_token],
        confidence=1 - args.label_smoothing).to(model_device)

    model = TransformerMT(
        src_vocab_size=corpus.src_vocab_size,
        tgt_vocab_size=corpus.tgt_vocab_size,
        joint_vocab_size=corpus.joint_vocab_size,
        share_embedding=args.share_embedding,
        share_projection_and_embedding=args.share_projection_and_embedding,
        src_pad_idx=corpus.src_word2idx[corpus.src_pad_token],
        tgt_pad_idx=corpus.tgt_word2idx[corpus.tgt_pad_token],
        tgt_sos_idx=corpus.tgt_word2idx[corpus.tgt_sos_token],
        tgt_eos_idx=corpus.tgt_word2idx[corpus.tgt_eos_token],
        positional_encoding=args.positional_encoding,
        emb_size=args.embedding_size,
        feed_forward_size=args.feedforward_size,
        num_of_layers=args.num_of_layers,
        num_of_heads=args.num_of_heads,
        train_max_seq_length=args.train_max_seq_length,
        infer_max_seq_length=args.infer_max_seq_length,
        infer_max_seq_length_mode=args.infer_max_seq_length_mode,
        batch_size=corpus.batch_size,
        update_decay=args.update_decay,
        embedding_dropout_prob=1 - args.embedding_keep_prob,
        attention_dropout_prob=1 - args.attention_keep_prob,
        feedforward_dropout_prob=1 - args.feedforward_keep_prob,
        residual_dropout_prob=1 - args.residual_keep_prob,
        activate_function_name=args.activate_function_name,
        emb_norm_clip=args.emb_norm_clip,
        emb_norm_clip_type=args.emb_norm_clip_type,
        layer_norm_pre=args.layer_norm_pre,
        layer_norm_post=args.layer_norm_post,
        layer_norm_encoder_start=args.layer_norm_encoder_start,
        layer_norm_encoder_end=args.layer_norm_encoder_end,
        layer_norm_decoder_start=args.layer_norm_decoder_start,
        layer_norm_decoder_end=args.layer_norm_decoder_end,
        prefix=args.file_prefix,
        pretrained_src_emb=args.pretrained_src_emb,
        pretrained_tgt_emb=args.pretrained_tgt_emb,
        pretrained_src_eos=args.pretrained_src_eos,
        pretrained_tgt_eos=args.pretrained_tgt_eos,
        src_vocab=corpus.src_word2idx,
        tgt_vocab=corpus.tgt_word2idx,
        criterion=criterion)

    print('*' * 80)
    print(model)
    stats.log_to_file(model.__repr__())
    stats.log_to_file('*' * 80)
    print('*' * 80)
    stats.log_to_file(model.model_parameters_statistic())

    torch.cuda.empty_cache()

    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    model.init_parameters()
    model.to(model_device)

    if len(device_idx) > 1:
        print('Multi-gpu is activated, totally %d gpus ... ' % len(device_idx))

    if args.optimizer == 'Adam' or args.optimizer == 'AMSGrad':
        optimizer = Adam(model.parameters(),
                         lr=0.0001,
                         betas=args.adam_betas,
                         eps=1e-9,
                         weight_decay=args.l2_scale,
                         amsgrad=args.optimizer == 'AMSGrad')
    else:
        optimizer = SGD(model.parameters(),
                        lr=0.0001,
                        weight_decay=args.l2_scale)

    if len(device_idx) == 1:
        trainer = TrainerSingleDevice(
            model=model,
            corpus=corpus,
            optimizer=optimizer,
            stats=stats,
            bleu=bleu,
            tgt_character_level=args.tgt_character_level,
            buffer_every_steps=args.buffer_every_steps,
            report_every_steps=args.report_every_steps,
            save_every_steps=args.save_every_steps,
            eval_every_steps=args.eval_every_steps,
            num_of_steps=args.num_of_steps,
            eval_type=args.eval_type,
            processed_steps=args.processed_steps,
            learning_rate_schedule=args.learning_rate_schedule,
            update_decay=args.update_decay,
            batch_capacity=args.batch_capacity,
            max_save_models=args.max_save_models,
            beam_size=args.beam_size,
            decoding_alpha=args.decoding_alpha,
            grad_norm_clip=args.grad_norm_clip,
            grad_norm_clip_type=args.grad_norm_clip_type,
            annotate=args.annotate,
            device=model_device,
            sample_times=args.sample_times,
            exponential_value=args.exponential_value,
            normalized_cl_factors=args.normalized_cl_factors)
    else:
        trainer = TrainerMultiDevice(
            model=model,
            corpus=corpus,
            optimizer=optimizer,
            stats=stats,
            bleu=bleu,
            tgt_character_level=args.tgt_character_level,
            buffer_every_steps=args.buffer_every_steps,
            report_every_steps=args.report_every_steps,
            save_every_steps=args.save_every_steps,
            eval_every_steps=args.eval_every_steps,
            num_of_steps=args.num_of_steps,
            eval_type=args.eval_type,
            processed_steps=args.processed_steps,
            learning_rate_schedule=args.learning_rate_schedule,
            update_decay=args.update_decay,
            batch_capacity=args.batch_capacity,
            max_save_models=args.max_save_models,
            beam_size=args.beam_size,
            decoding_alpha=args.decoding_alpha,
            grad_norm_clip=args.grad_norm_clip,
            grad_norm_clip_type=args.grad_norm_clip_type,
            num_of_workers=args.num_of_workers,
            annotate=args.annotate,
            device_idxs=device_idx,
            training_batch_chunks_ratio=args.training_batch_chunks_ratio,
            sample_times=args.sample_times,
            exponential_value=args.exponential_value,
            normalized_cl_factors=args.normalized_cl_factors)

    if len(args.retrain_model) != 0:
        trainer.retrain_model(
            retrain_model=args.retrain_model,
            processed_steps=args.processed_steps)
    try:
        trainer.run()
        print('Logging stats to file ... ')
        stats.log_to_file('*' * 80)
    except KeyboardInterrupt:
        print('Keyboard interrupted. Logging stats to file ... ')
    except RuntimeError as e:
        if 'CUDA out of memory' in e.args[0]:
            print('Try to decrease batch_capacity or increase update_decay.')
        else:
            raise e
    finally:
        stats.train_stats_to_file()
        stats.valid_stats_to_file()
        trainer.release()

    if args.eval_type == 'acc':
        sorted_values = sorted(stats.valid_acc.items(), key=lambda d: d[1], reverse=True)
    elif args.eval_type == 'xent':
        sorted_values = sorted(stats.valid_loss.items(), key=lambda d: d[1])
    else:
        sorted_values = sorted(stats.valid_bleu.items(), key=lambda d: d[1], reverse=True)

    stats.log_to_file('Model performances (%s): ' % args.eval_type)
    print('Model performances (%s): ' % args.eval_type)
    for (step, value) in sorted_values:
        print('%6d\t%8f' % (step, value))
        stats.log_to_file('%d\t%8f' % (step, value))
