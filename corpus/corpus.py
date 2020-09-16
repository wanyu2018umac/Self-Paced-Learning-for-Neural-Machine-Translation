import torch
import numpy
import math
import time
import os
from shutil import copy
from torch.multiprocessing import Queue, Pool, Process, Manager
from multiprocessing.managers import ValueProxy
from torch.nn.utils.rnn import pad_sequence
from threading import Thread, Lock, current_thread

from utils.bpe import BytePairEncoding
from utils.stats import Stats


class Corpus:
    def __init__(self, prefix: str,
                 corpus_source_train: str, corpus_source_valid: str, corpus_source_test: [str],
                 corpus_target_train: str, corpus_target_valid: str, corpus_target_test: [str],
                 src_pad_token: str, src_unk_token: str,
                 src_sos_token: str, src_eos_token: str,
                 tgt_pad_token: str, tgt_unk_token: str,
                 tgt_sos_token: str, tgt_eos_token: str,
                 share_embedding: bool,
                 bpe_suffix_token: str,
                 bpe_src: bool, bpe_tgt: bool,
                 min_seq_length: int,
                 max_seq_length: int,
                 length_merging_mantissa_bits: int,
                 batch_size: int,
                 logger: Stats,
                 num_of_workers: int,
                 num_of_steps: int,
                 batch_capacity: int,
                 train_buffer_size: int,
                 train_prefetch_size: int,
                 device: torch.device):

        self.prefix = prefix
        self.corpus_source_train_name = corpus_source_train
        self.corpus_source_valid_name = corpus_source_valid
        self.corpus_source_test_name = corpus_source_test
        self.corpus_target_train_name = corpus_target_train
        self.corpus_target_valid_name = corpus_target_valid
        self.corpus_target_test_name = corpus_target_test

        self.src_pad_token = src_pad_token
        self.src_unk_token = src_unk_token
        self.src_sos_token = src_sos_token
        self.src_eos_token = src_eos_token

        self.tgt_pad_token = tgt_pad_token
        self.tgt_unk_token = tgt_unk_token
        self.tgt_sos_token = tgt_sos_token
        self.tgt_eos_token = tgt_eos_token

        self.src_sos = True if self.src_sos_token != '' and self.src_sos_token else False
        self.src_eos = True if self.src_eos_token != '' and self.src_eos_token else False
        self.tgt_sos = True if self.tgt_sos_token != '' and self.tgt_sos_token else False
        self.tgt_eos = True if self.tgt_eos_token != '' and self.tgt_eos_token else False

        self.share_embedding = share_embedding

        self.src_special_tokens = [self.src_pad_token, self.src_unk_token, self.src_sos_token, self.src_eos_token]
        self.tgt_special_tokens = [self.tgt_pad_token, self.tgt_unk_token, self.tgt_sos_token, self.tgt_eos_token]
        self.joint_special_tokens = list()

        if self.share_embedding:
            for src_token, tgt_token in zip(self.src_special_tokens, self.tgt_special_tokens):
                if src_token == '' and tgt_token == '':
                    continue
                elif src_token == tgt_token:
                    self.joint_special_tokens.append(src_token)
                elif src_token == '' or tgt_token == '':
                    self.joint_special_tokens.append(src_token) if src_token != '' \
                        else self.joint_special_tokens.append(tgt_token)
                else:
                    self.joint_special_tokens.append(src_token)
                    self.joint_special_tokens.append(tgt_token)

        self.joint_special_tokens = list(
            d[1] for d in filter(lambda d: d[1] != '', enumerate(self.joint_special_tokens)))
        self.src_special_tokens = list(d[1] for d in filter(lambda d: d[1] != '', enumerate(self.src_special_tokens)))
        self.tgt_special_tokens = list(d[1] for d in filter(lambda d: d[1] != '', enumerate(self.tgt_special_tokens)))

        for idx in range(len(self.src_special_tokens) - 1, -1, -1):
            if self.src_special_tokens.count(self.src_special_tokens[idx]) > 1:
                self.src_special_tokens.pop(idx)
            else:
                idx -= 1

        for idx in range(len(self.tgt_special_tokens) - 1, -1, -1):
            if self.tgt_special_tokens.count(self.tgt_special_tokens[idx]) > 1:
                self.tgt_special_tokens.pop(idx)
            else:
                idx -= 1

        self.logger = logger
        self.device = device
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.num_of_workers = num_of_workers
        self.num_of_steps = num_of_steps
        self.batch_capacity = batch_capacity
        self.train_buffer_size = train_buffer_size
        self.train_prefetch_size = train_prefetch_size
        self.manager = Manager()

        self.long_seq_idx = set()
        self.short_seq_idx = set()
        self.empty_seq_idx = set()
        self.escaped_seq_idx = set()

        self.length_merging_mantissa_bits = length_merging_mantissa_bits
        self.batch_size = batch_size

        self.bpe_suffix_token = bpe_suffix_token

        self.bpe_trans_table = {}
        self.byte_pair_handler_src = None
        self.byte_pair_handler_tgt = None

        self.bpe_src = bpe_src
        self.bpe_tgt = bpe_tgt

        self.batches_source_train = Queue()
        self.batches_source_train_length = Queue()
        self.batches_source_valid = dict()
        self.batches_source_valid_length = dict()
        self.batches_source_test = dict()
        self.batches_target_train = Queue()
        self.batches_target_train_length = Queue()
        self.batches_target_valid = dict()
        self.batches_target_valid_length = dict()
        self.batches_target_test = dict()

        self.corpus_source_train_numerate = dict()
        self.corpus_target_train_numerate = dict()
        self.corpus_source_valid_numerate = dict()
        self.corpus_target_valid_numerate = dict()
        self.corpus_source_test = dict()
        self.corpus_target_test = dict()
        self.corpus_source_train_num_of_lines = 0
        self.corpus_target_train_num_of_lines = 0

        self.corpus_source_train_length = torch.empty((0,))
        self.corpus_target_train_length = torch.empty((0,))
        self.corpus_source_valid_length = torch.empty((0,))
        self.corpus_target_valid_length = torch.empty((0,))

        self.corpus_train_lengths_stats = dict()

        self.source_frequency = dict()
        self.target_frequency = dict()
        self.joint_frequency = dict()

        self.src_word2idx = dict()
        self.src_idx2word = dict()
        self.tgt_word2idx = dict()
        self.tgt_idx2word = dict()
        self.joint_word2idx = dict()
        self.joint_idx2word = dict()
        self.src_vocab_size = 0
        self.tgt_vocab_size = 0
        self.joint_vocab_size = 0

        self.num_of_multi_refs = 1
        self.num_of_made_segments = self.manager.Value('int', 0)
        self.num_of_made_batches = self.manager.Value('int', 0)
        self.num_of_trained_batches = self.manager.Value('int', 0)
        self.shuffled_times = self.manager.Value('int', 0)
        self.train_pool = self.manager.list()

        self.pool_lock = self.manager.Lock()
        self.batch_lock = self.manager.Lock()

        self.running = self.manager.Value('bool', True)

        return

    def train_file_stats(self):
        with open(self.prefix + self.corpus_source_train_name, mode='r', encoding='utf-8') as f1, \
                open(self.prefix + self.corpus_target_train_name, mode='r', encoding='utf-8') as f2:
            src_len = len(list(f1))
            tgt_len = len(list(f2))

            print('Corpus source train file:', self.prefix + self.corpus_source_train_name, src_len, 'lines.')
            print('Corpus target train file:', self.prefix + self.corpus_target_train_name, tgt_len, 'lines.')

            if src_len != tgt_len:
                raise ValueError('Corpus train files %s, %s do not have same number of parallel sentences.'
                                 % (self.prefix + self.corpus_source_train_name, self.prefix + self.corpus_target_train_name))

        self.corpus_source_train_num_of_lines = src_len
        self.corpus_target_train_num_of_lines = tgt_len

        with open(self.prefix + self.corpus_source_train_name, mode='r', encoding='utf-8') as f1, \
                open(self.prefix + self.corpus_target_train_name, mode='r', encoding='utf-8') as f2:

            size_of_shard = self.corpus_source_train_num_of_lines // self.num_of_workers + 1
            bounds = [0] + list(size_of_shard * i for i in range(1, self.num_of_workers + 1))
            queue = Queue(maxsize=self.num_of_workers)
            pool = list()

            for i in range(0, self.num_of_workers):
                p = Process(target=_corpus_length_limit_worker,
                            args=(self.prefix + self.corpus_source_train_name,
                                  self.prefix + self.corpus_target_train_name,
                                  bounds[i], bounds[i + 1],
                                  self.min_seq_length, self.max_seq_length, queue))
                pool.append(p)

            for p in pool:
                p.start()

            for i in range(0, self.num_of_workers):
                short_idxs, long_idxs, empty_idxs = queue.get()
                self.short_seq_idx.update(short_idxs)
                self.long_seq_idx.update(long_idxs)
                self.empty_seq_idx.update(empty_idxs)

            for p in pool:
                p.join()
            for p in pool:
                p.close()

        self.escaped_seq_idx = self.short_seq_idx | self.long_seq_idx | self.empty_seq_idx

        print('Corpus stats on lengths:')
        print('\tShorter than %d: %d' % (self.min_seq_length, len(self.short_seq_idx)))
        print('\tLonger than %d: %d' % (self.max_seq_length, len(self.long_seq_idx)))
        print('\tEmpty sequences: %d' % len(self.empty_seq_idx))
        print('\tTotal escaped sequences: %d' % (len(self.escaped_seq_idx)))
        print('*' * 80)

        return

    def valid_file_stats(self):
        with open(self.prefix + self.corpus_source_valid_name, mode='r', encoding='utf-8') as f1, \
                open(self.prefix + self.corpus_target_valid_name, mode='r', encoding='utf-8') as f2:
            print('Corpus source valid file:', self.prefix + self.corpus_source_valid_name, len(list(f1)), 'lines.')
            print('Corpus target valid file:', self.prefix + self.corpus_target_valid_name, len(list(f2)), 'lines.')

            if len(list(f1)) != len(list(f2)):
                raise ValueError('Corpus valid files %s, %s do not have same numer of parallel sentences.'
                                 % (self.corpus_source_valid_name, self.corpus_target_valid_name))

        return

    def test_file_stats(self):
        for src_path, tgt_path in zip(self.corpus_source_test_name, self.corpus_target_test_name):
            with open(self.prefix + src_path, mode='r', encoding='utf-8') as f1, \
                    open(self.prefix + tgt_path, mode='r', encoding='utf-8') as f2:
                print('Corpus source test file:', self.prefix + src_path, len(list(f1)), 'lines.')
                print('Corpus target test file:', self.prefix + tgt_path, len(list(f2)), 'lines.')

                if len(list(f1)) != len(list(f2)):
                    raise ValueError('Corpus test files %s, %s do not have same numer of parallel sentences.'
                                     % (self.prefix + src_path, self.prefix + tgt_path))

        return

    def build_vocab(self, src_vocab_size: int, tgt_vocab_size: int,
                    src_vocab_path: str = '', tgt_vocab_path: str = '', joint_vocab_path: str = ''):

        print('Special tokens:')
        print('\tSource: pad: %s, unk: %s, sos: %s, eos: %s'
              % (self.src_pad_token, self.src_unk_token,
                 None if self.src_sos_token == '' else self.src_sos_token,
                 None if self.src_eos_token == '' else self.src_eos_token))
        print('\tTarget: pad: %s, unk: %s, sos: %s, eos: %s'
              % (self.tgt_pad_token, self.tgt_unk_token,
                 None if self.tgt_sos_token == '' else self.tgt_sos_token,
                 None if self.tgt_eos_token == '' else self.tgt_eos_token))

        if self.logger:
            self.logger.log_to_file('*' * 80)
            self.logger.log_to_file('Special tokens:')
            self.logger.log_to_file('\tSource: pad: %s, unk: %s, sos: %s, eos: %s'
                                    % (self.src_pad_token, self.src_unk_token, self.src_sos_token, self.src_eos_token))
            self.logger.log_to_file('\tTarget: pad: %s, unk: %s, sos: %s, eos: %s'
                                    % (self.tgt_pad_token, self.tgt_unk_token, self.tgt_sos_token, self.tgt_eos_token))
            self.logger.log_to_file('*' * 80)

        print('Vocab mode: ' + 'share' if self.share_embedding else 'Vocab mode: ' + 'independent')

        if src_vocab_path and src_vocab_path != '' and tgt_vocab_path and tgt_vocab_path != '':
            self.src_word2idx = torch.load(src_vocab_path)
            self.tgt_word2idx = torch.load(tgt_vocab_path)

            self.src_idx2word.update(zip(self.src_word2idx.values(), self.src_word2idx.keys()))
            self.tgt_idx2word.update(zip(self.tgt_word2idx.values(), self.tgt_word2idx.keys()))

            print('Load source vocab from ' + src_vocab_path + ':', len(self.src_word2idx))
            print('Load target vocab from ' + tgt_vocab_path + ':', len(self.tgt_word2idx))

            if self.logger:
                copy(src_vocab_path, self.logger.fold_name + '/src_vocab.pt')
                copy(tgt_vocab_path, self.logger.fold_name + '/tgt_vocab.pt')
        else:
            size_of_shard = self.corpus_source_train_num_of_lines // self.num_of_workers + 1
            bounds = [0] + list(size_of_shard * i for i in range(1, self.num_of_workers + 1))
            queue = Queue(maxsize=self.num_of_workers)
            pool = list()

            for i in range(0, self.num_of_workers):
                p = Process(target=_corpus_vocab_worker,
                            args=(self.prefix + self.corpus_source_train_name,
                                  self.prefix + self.corpus_target_train_name,
                                  bounds[i], bounds[i + 1], self.escaped_seq_idx, queue))
                pool.append(p)

            for p in pool:
                p.start()

            for i in range(0, self.num_of_workers):
                d1, d2 = queue.get()
                self.source_frequency.update(zip(d1.keys(),
                                                 map(lambda w: self.source_frequency.get(w, 0) + d1[w], d1.keys())))
                self.target_frequency.update(zip(d2.keys(),
                                                 map(lambda w: self.target_frequency.get(w, 0) + d2[w], d2.keys())))

            for p in pool:
                p.join()
            for p in pool:
                p.close()

            original_vocab_size = len(self.source_frequency)
            vocab_source_freq_sorted = sorted(self.source_frequency.items(), key=lambda d: d[1], reverse=True)
            vocab_source_tobe = list(d[0] for d in vocab_source_freq_sorted)

            for idx, special_token in enumerate(self.src_special_tokens):
                if special_token in vocab_source_tobe:
                    raise KeyError('Special token %s conflicts in source vocab.' % special_token)
                self.src_word2idx[special_token] = idx
                self.src_idx2word[idx] = special_token

            vocab_source_tobe = vocab_source_tobe[:src_vocab_size]
            vocab_source_offset = len(self.src_word2idx)
            self.src_word2idx.update(zip(vocab_source_tobe, range(vocab_source_offset,
                                                                  len(vocab_source_tobe) + vocab_source_offset)))
            self.src_idx2word.update(enumerate(vocab_source_tobe, start=vocab_source_offset))
            print('Source vocab:', original_vocab_size, '->', len(self.src_word2idx))

            original_vocab_size = len(self.target_frequency)
            vocab_target_freq_sorted = sorted(self.target_frequency.items(), key=lambda d: d[1], reverse=True)
            vocab_target_tobe = list(d[0] for d in vocab_target_freq_sorted)

            for idx, special_token in enumerate(self.tgt_special_tokens):
                if special_token in vocab_target_tobe:
                    raise KeyError('Special token %s conflicts in target vocab.' % special_token)
                self.tgt_word2idx[special_token] = idx
                self.tgt_idx2word[idx] = special_token

            vocab_target_tobe = vocab_target_tobe[:tgt_vocab_size]
            vocab_target_offset = len(self.tgt_word2idx)
            self.tgt_word2idx.update(zip(vocab_target_tobe, range(vocab_target_offset,
                                                                  len(vocab_target_tobe) + vocab_target_offset)))
            self.tgt_idx2word.update(enumerate(vocab_target_tobe, start=vocab_target_offset))
            print('Target vocab:', original_vocab_size, '->', len(self.tgt_word2idx))

        if self.share_embedding:
            if joint_vocab_path and joint_vocab_path != '':
                self.joint_word2idx = torch.load(joint_vocab_path)
                self.joint_idx2word = dict(zip(self.joint_word2idx.values(), self.joint_word2idx.keys()))
                print('Load joint vocab from ' + joint_vocab_path + ':', len(self.joint_word2idx))
            else:
                vocab_comple = set(self.src_word2idx.keys()) - set(self.tgt_word2idx.keys())

                print('Merging source vocab and target vocab, %d words not seen in target vocab.' % len(vocab_comple))
                self.joint_word2idx.update(self.tgt_word2idx.items())
                self.joint_idx2word.update(self.tgt_idx2word.items())

                for special_token in self.joint_special_tokens:
                    if special_token not in self.joint_word2idx.keys():
                        self.joint_word2idx[special_token] = len(self.joint_word2idx)
                        self.joint_idx2word[len(self.joint_idx2word)] = special_token
                        vocab_comple.remove(special_token)

                vocab_comple_tobe = sorted(vocab_comple, key=lambda k: self.src_idx2word.get(k, 0), reverse=True)
                vocab_joint_offset = len(self.joint_word2idx)
                self.joint_idx2word.update(enumerate(vocab_comple, start=vocab_joint_offset))
                self.joint_word2idx.update(zip(vocab_comple_tobe, range(vocab_joint_offset,
                                                                        len(vocab_comple_tobe) + vocab_joint_offset)))

                print('Merged vocab applied, source and target vocabs are merged: %d' % len(self.joint_word2idx))

                self.src_word2idx.update(self.joint_word2idx)
                self.src_idx2word.update(self.joint_idx2word)

                self.tgt_word2idx.update(self.joint_word2idx)
                self.tgt_idx2word.update(self.joint_idx2word)

        self.src_vocab_size = len(self.src_word2idx)
        self.tgt_vocab_size = len(self.tgt_word2idx)
        self.joint_vocab_size = len(self.joint_word2idx)

        print('Source vocab:', self.src_vocab_size)
        print('Target vocab:', self.tgt_vocab_size)
        print('Joint vocab:', self.joint_vocab_size)

        self.byte_pair_handler_src = BytePairEncoding(bpe_suffix_token=self.bpe_suffix_token,
                                                      vocab=self.src_word2idx)
        self.byte_pair_handler_tgt = BytePairEncoding(bpe_suffix_token=self.bpe_suffix_token,
                                                      vocab=self.tgt_word2idx)

        return

    def src_special_tokens_handler(self, tokens: [str]):
        if self.src_sos:
            tokens = [self.src_sos_token] + tokens
        if self.src_eos:
            tokens = tokens + [self.src_eos_token]
        return tokens

    def tgt_special_tokens_handler(self, tokens: [str]):
        if self.tgt_sos:
            tokens = [self.tgt_sos_token] + tokens
        if self.tgt_eos:
            tokens = tokens + [self.tgt_eos_token]
        return tokens

    def corpus_numerate_train(self, src_corpus_enumerate_path: str = '', tgt_corpus_enumerate_path: str = ''):
        if src_corpus_enumerate_path and src_corpus_enumerate_path != '' and \
                tgt_corpus_enumerate_path and tgt_corpus_enumerate_path != '':
            print('Load source enumerate from', src_corpus_enumerate_path)
            self.corpus_source_train_numerate = torch.load(src_corpus_enumerate_path)

            print('Load target enumerate from', tgt_corpus_enumerate_path)
            self.corpus_target_train_numerate = torch.load(tgt_corpus_enumerate_path)

            self.corpus_source_train_length = torch.Tensor(
                list(len(x) for x in self.corpus_source_train_numerate.values())).long()
            self.corpus_target_train_length = torch.Tensor(
                list(len(x) for x in self.corpus_target_train_numerate.values())).long()
        else:
            size_of_shard = self.corpus_source_train_num_of_lines // self.num_of_workers + 1
            bounds = [0] + list(size_of_shard * i for i in range(1, self.num_of_workers + 1))
            queue = Queue(maxsize=self.num_of_workers)
            pool = list()

            for i in range(0, self.num_of_workers):
                p = Process(target=_corpus_numerate_worker,
                            args=(self.prefix + self.corpus_source_train_name,
                                  self.prefix + self.corpus_target_train_name,
                                  bounds[i], bounds[i + 1], self.escaped_seq_idx,
                                  self.src_word2idx, self.tgt_word2idx,
                                  self.src_sos_token, self.src_eos_token, self.src_unk_token,
                                  self.tgt_sos_token, self.tgt_eos_token, self.tgt_unk_token,
                                  queue))
                pool.append(p)

            for p in pool:
                p.start()

            source_lengths = dict()
            target_lengths = dict()

            for i in range(0, self.num_of_workers):
                d1, d2, l1, l2 = queue.get()
                self.corpus_source_train_numerate.update(d1)
                self.corpus_target_train_numerate.update(d2)
                source_lengths.update(l1)
                target_lengths.update(l2)

            for p in pool:
                p.join()
            for p in pool:
                p.close()

            self.corpus_source_train_numerate = dict(
                sorted(self.corpus_source_train_numerate.items(), key=lambda d: d[0]))
            self.corpus_target_train_numerate = dict(
                sorted(self.corpus_target_train_numerate.items(), key=lambda d: d[0]))
            source_lengths = dict(sorted(source_lengths.items(), key=lambda d: d[0]))
            target_lengths = dict(sorted(target_lengths.items(), key=lambda d: d[0]))
            self.corpus_source_train_length = torch.Tensor(list(source_lengths.values())).long()
            self.corpus_target_train_length = torch.Tensor(list(target_lengths.values())).long()

        # self.corpus_source_train_numerate = dict((k, v.pin_memory())
        #                                          for (k, v) in self.corpus_source_train_numerate.items())
        # self.corpus_target_train_numerate = dict((k, v.pin_memory())
        #                                          for (k, v) in self.corpus_target_train_numerate.items())

        print('Train files are enumerated.')
        return

    def corpus_numerate_valid(self):
        with open(self.prefix + self.corpus_source_valid_name, mode='r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                tokens = self.src_special_tokens_handler(line.split())
                if self.bpe_src:
                    tokens = self.byte_pair_handler_src.words2subwords(tokens)
                enum_seq = torch.Tensor(numpy.array(
                    list(self.src_word2idx[word]
                         if word in self.src_word2idx.keys() else self.src_word2idx[self.src_unk_token]
                         for word in tokens))).long()
                if enum_seq.numel() == 0:
                    continue
                self.corpus_source_valid_numerate[len(self.corpus_source_valid_numerate)] = enum_seq
            self.corpus_source_valid_length = \
                torch.Tensor(list(len(seq) for seq in self.corpus_source_valid_numerate.values())).long()

        with open(self.prefix + self.corpus_target_valid_name, mode='r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                tokens = self.tgt_special_tokens_handler(line.split())
                if self.bpe_tgt:
                    tokens = self.byte_pair_handler_tgt.words2subwords(tokens)
                enum_seq = torch.Tensor(numpy.array(
                    list(self.tgt_word2idx[word]
                         if word in self.tgt_word2idx.keys() else self.tgt_word2idx[self.tgt_unk_token]
                         for word in tokens))).long()
                if enum_seq.numel() == 0:
                    continue
                self.corpus_target_valid_numerate[len(self.corpus_target_valid_numerate)] = enum_seq
            self.corpus_target_valid_length = \
                torch.Tensor(list(len(seq) for seq in self.corpus_target_valid_numerate.values())).long()
        print('Valid files are enumerated.')

        return

    def corpus_numerate_test(self):
        for src_path, tgt_path in zip(self.corpus_source_test_name, self.corpus_target_test_name):
            with open(self.prefix + src_path, mode='r', encoding='utf-8') as f:
                temp_dict = {}
                for i, line in enumerate(f):
                    tokens = self.src_special_tokens_handler(line.split())
                    if len(tokens) == 0:
                        continue
                    if self.bpe_src:
                        temp_dict[i] = self.byte_pair_handler_src.words2subwords(tokens)
                    else:
                        temp_dict[i] = line.split()
                self.corpus_source_test[len(self.corpus_source_test)] = temp_dict

            with open(self.prefix + tgt_path, mode='r', encoding='utf-8') as f:
                temp_dict = {}
                for i, line in enumerate(f):
                    tokens = self.tgt_special_tokens_handler(line.split())
                    if len(tokens) == 0:
                        continue
                    if self.bpe_tgt:
                        temp_dict[i] = self.byte_pair_handler_tgt.words2subwords(line.split())
                    else:
                        temp_dict[i] = line.split()
                self.corpus_target_test[len(self.corpus_target_test)] = temp_dict

        print('Test files are enumerated.')
        return

    def valid_batch_making(self, batch_size: int):
        num_of_steps = 0
        self.num_of_multi_refs = len(self.corpus_target_valid_numerate) // len(self.corpus_source_valid_numerate)

        if self.num_of_multi_refs > 1:
            print('Applying %d multi references when validating.' % self.num_of_multi_refs)
            if len(self.corpus_target_valid_numerate) % len(self.corpus_source_valid_numerate) != 0:
                raise Exception('Multi references cannot be applied due to undivisible target valid examples.')

        sorted_pool = sorted((zip(range(0, len(self.corpus_source_valid_numerate)),
                                  self.corpus_source_valid_length,
                                  self.corpus_target_valid_length)),
                             key=lambda d: (d[1], d[2]), reverse=True)
        pool = list(d[0] for d in sorted_pool)

        num_of_batches = math.ceil(len(pool) / batch_size)
        num_of_examples = len(pool)

        while num_of_steps < num_of_batches:
            idx = pool[num_of_steps * batch_size: (num_of_steps + 1) * batch_size]
            batch_size = len(idx)

            self.batches_source_valid[num_of_steps] = \
                pad_sequence([self.corpus_source_valid_numerate[x] for x in idx],
                             batch_first=True, padding_value=self.src_word2idx[self.src_pad_token]).contiguous()
            self.batches_source_valid_length[num_of_steps] = \
                torch.Tensor(list(len(self.corpus_source_valid_numerate[x]) for x in idx)).int()

            idx = numpy.array(idx)

            for multi_ref_step in range(0, self.num_of_multi_refs):
                idx_ref = idx + multi_ref_step * num_of_examples
                self.batches_target_valid[num_of_steps + multi_ref_step * num_of_batches] = \
                    pad_sequence([(self.corpus_target_valid_numerate[x]) for x in idx_ref],
                                 batch_first=True, padding_value=self.tgt_word2idx[self.tgt_pad_token]).contiguous()
                self.batches_target_valid_length[num_of_steps + multi_ref_step * num_of_batches] = \
                    torch.Tensor(list(len(self.corpus_target_valid_numerate[x]) for x in idx)).int()

            num_of_steps += 1

        print('Valid batches: 0 -> %d' % (len(self.batches_source_valid)))
        return

    def corpus_train_lengths_sorting(self):
        if self.length_merging_mantissa_bits < 0:
            print('Mantissa bits for merging examples disabled.')
            src_lengths = self.corpus_source_train_length.numpy().tolist()
            tgt_lengths = self.corpus_target_train_length.numpy().tolist()
            all_lengths_list = list((x, y) for x, y in zip(src_lengths, tgt_lengths))
        else:
            print('Mantissa bits for merging examples enabled by %d.' % self.length_merging_mantissa_bits)

            src_lengths = self.corpus_source_train_length.clone()
            tgt_lengths = self.corpus_target_train_length.clone()

            bounds = list()
            x = self.min_seq_length
            while x < self.max_seq_length:
                bounds.append(x)
                x += 2 ** max(0, int(math.log(x, 2)) - self.length_merging_mantissa_bits)

            print('Lengths will be merged into buckets: %s' % bounds)

            for i in range(0, len(bounds) - 1):
                mask = src_lengths.gt(bounds[i]) & src_lengths.lt(bounds[i + 1])
                src_lengths.masked_fill_(mask, bounds[i])
            for i in range(0, len(bounds) - 1):
                mask = tgt_lengths.gt(bounds[i]) & tgt_lengths.lt(bounds[i + 1])
                tgt_lengths.masked_fill_(mask, bounds[i])

            src_lengths = src_lengths.numpy().tolist()
            tgt_lengths = tgt_lengths.numpy().tolist()
            all_lengths_list = list((x, y) for x, y in zip(src_lengths, tgt_lengths))

        for idx, lengths_pair in enumerate(all_lengths_list):
            if lengths_pair in self.corpus_train_lengths_stats.keys():
                self.corpus_train_lengths_stats[lengths_pair] += [idx]
            else:
                self.corpus_train_lengths_stats[lengths_pair] = [idx]

        print('Train source-target lengths %d pairs in total' % (len(self.corpus_train_lengths_stats)))
        return

    def invoke_train_batches_making(self):
        with self.batch_lock:
            print('Making batches process activated ... ')
            process = Process(target=_batch_prefetch,
                              args=((self.src_word2idx[self.src_pad_token], self.tgt_word2idx[self.tgt_pad_token],
                                     self.running,
                                     self.num_of_made_batches, self.num_of_trained_batches, self.num_of_steps,
                                     self.train_pool, self.train_buffer_size, self.train_prefetch_size,
                                     self.pool_lock, self.batch_lock,
                                     self.corpus_source_train_numerate, self.corpus_target_train_numerate,
                                     self.corpus_source_train_length, self.corpus_target_train_length,
                                     self.batches_source_train, self.batches_target_train,
                                     self.batches_source_train_length, self.batches_target_train_length,
                                     self.logger)))

        return process

    def invoke_train_segments_making(self):
        with self.pool_lock:
            print('Making segments process activated ... ')
            process = Process(target=_data_segment,
                              args=((self.batch_capacity, self.num_of_made_segments, self.num_of_steps,
                                     self.running,
                                     self.corpus_train_lengths_stats,
                                     self.pool_lock, self.batch_lock,
                                     self.train_pool, self.shuffled_times, self.logger)))

        return process

    def get_train_batches(self, num_of_batches: int):
        batches = list()

        if self.num_of_made_batches.value - self.num_of_trained_batches.value >= num_of_batches:
            with self.batch_lock:
                source = list(x.to(self.device) for x in self.batches_source_train.get())
                target = list(x.to(self.device) for x in self.batches_target_train.get())
                source_lengths = list(x.to(self.device) for x in self.batches_source_train_length.get())
                target_lengths = list(x.to(self.device) for x in self.batches_target_train_length.get())
                self.num_of_trained_batches.value += num_of_batches

            batches = zip(source, target, source_lengths, target_lengths)
        else:
            time.sleep(1)

        return batches

    def get_valid_batches(self):
        num_of_batches = len(self.batches_source_valid)
        keys = self.batches_source_valid.keys()

        source = list(self.batches_source_valid[i].to(self.device) for i in keys)
        target = list(list(self.batches_target_valid[i].to(self.device)
                           for i in range(j, self.num_of_multi_refs * num_of_batches, num_of_batches))
                      for j in keys)
        src_lens = list(self.batches_source_valid_length[i].to(self.device) for i in keys)
        tgt_lens = list(list(self.batches_target_valid_length[i].to(self.device)
                             for i in range(j, self.num_of_multi_refs * num_of_batches, num_of_batches))
                        for j in keys)

        return source, target, src_lens, tgt_lens

    def get_valid_batches_for_translation(self):
        keys = self.batches_source_valid.keys()
        num_of_batches = len(keys)

        tgt_eos_idx = self.tgt_word2idx[self.tgt_eos_token]
        tgt_pad_idx = self.tgt_word2idx[self.tgt_pad_token]

        source = list(self.batches_source_valid[i].to(self.device) for i in keys)
        target = []

        for ref_no in range(0, self.num_of_multi_refs):
            target_ref = []
            for i in range(ref_no * num_of_batches, (ref_no + 1) * num_of_batches):
                target_now = self.batches_target_valid[i][:, 1:]
                target_now = target_now.masked_fill(target_now.eq(tgt_eos_idx), tgt_pad_idx)
                target_lengths = target_now.ne(self.tgt_word2idx[self.tgt_pad_token]).sum(dim=-1)
                target_temp = list(x[:l].cpu().numpy().tolist()
                                   for x, l in zip(torch.unbind(target_now, dim=0), target_lengths))
                target_ref += target_temp

            target.append(target_ref)

        return source, target

    def get_test_batches(self, num_of_test: int):
        src = self.corpus_source_test[num_of_test]
        tgt = self.corpus_target_test[num_of_test]
        num_of_batches = math.ceil(len(src) / self.batch_size)

        print('Source file:', self.prefix + self.corpus_source_test_name[num_of_test])
        print('Reference file:', self.prefix + self.corpus_target_test_name[num_of_test])

        self.num_of_multi_refs = len(tgt) // len(src)
        if self.num_of_multi_refs > 1:
            print('Applying %d multi references when validating.' % self.num_of_multi_refs)

            if len(tgt) % len(src) != 0:
                raise Exception('Multi references cannot be applied due to undivisible target valid examples.')

        src_seqs = list(torch.Tensor(list(self.src_word2idx[word]
                                          if word in self.src_word2idx.keys()
                                          else self.src_word2idx[self.src_unk_token]
                                          for word in source)).long().to(self.device)
                        for source in src.values())
        target_batches = []
        pool = sorted(enumerate(list(x.size(0) for x in src_seqs)), key=lambda d: d[1], reverse=True)
        order = list(d[0] for d in pool)
        ordered_src_seqs = list(src_seqs[x] for x in order)

        source_batches = list(pad_sequence(ordered_src_seqs[i * self.batch_size: (i + 1) * self.batch_size],
                                           batch_first=True, padding_value=self.src_word2idx[self.src_pad_token])
                              for i in range(0, num_of_batches))

        for i in range(0, self.num_of_multi_refs):
            seq = list(list(self.tgt_word2idx.get(word, self.tgt_word2idx[self.tgt_pad_token])
                            for word in tgt[k]) for k in range(i * len(src), (i + 1) * len(src)))
            ordered_tgt_seq = list(seq[x] for x in order)
            target_batches.append(ordered_tgt_seq)

        return source_batches, target_batches, order

    def tensor2sentence_source(self, inputs):
        if isinstance(inputs, torch.Tensor):
            if inputs.ndimension() == 1:
                inputs.unsqueeze_(dim=0)
            inputs = inputs.cpu().numpy().tolist()

        if not isinstance(inputs, list):
            raise TypeError

        outputs = list(list(self.src_idx2word[x] for x in example) for example in inputs)
        if self.bpe_src:
            outputs = list(self.byte_pair_handler_src.subwords2words[x] for x in outputs)
        outputs = list(' '.join(x) for x in outputs)
        return outputs

    def tensor2sentence_target(self, inputs):
        if isinstance(inputs, torch.Tensor):
            if inputs.ndimension() == 1:
                inputs.unsqueeze_(dim=0)
            inputs = inputs.cpu().numpy().tolist()

        if not isinstance(inputs, list):
            raise TypeError

        outputs = list(list(self.tgt_idx2word[x] for x in example) for example in inputs)
        if self.bpe_tgt:
            outputs = list(self.byte_pair_handler_tgt.subwords2words[x] for x in outputs)
        outputs = list(' '.join(x) for x in outputs)
        return outputs

    # def __iter__(self):
    #     return self.get_train_batches(num_of_batches=10, batch_capacity=8192)
    #
    # def __len__(self):
    #     return len(self.train_pool)


def _special_tokens_handler(tokens: [str], sos_token: str, eos_token: str):
    if sos_token and eos_token:
        tokens = [sos_token] + tokens + [eos_token]
    elif sos_token:
        tokens = [sos_token] + tokens
    elif eos_token:
        tokens = tokens + [eos_token]

    return tokens


def _corpus_length_limit_worker(src_file_path: str, tgt_file_path: str,
                                idx_start: int, idx_end: int,
                                min_seq_length: int, max_seq_length: int,
                                queue: Queue):
    short_idxs = dict()
    long_idxs = dict()
    empty_idxs = dict()

    with open(src_file_path, mode='r', encoding='utf-8') as f1, open(tgt_file_path, mode='r', encoding='utf-8') as f2:
        for idx, (src, tgt) in enumerate(zip(f1, f2)):
            if idx < idx_start:
                continue
            if idx >= idx_end:
                break

            x1 = len(src.split())
            x2 = len(tgt.split())

            if x1 < min_seq_length or x2 < min_seq_length:
                short_idxs[idx] = 0
            if x1 > max_seq_length or x2 > max_seq_length:
                long_idxs[idx] = 0
            if x1 == 0 or x2 == 0:
                empty_idxs[idx] = 0

    queue.put((set(short_idxs.keys()), set(long_idxs.keys()), set(empty_idxs.keys())))
    return


def _corpus_vocab_worker(src_file_path: str, tgt_file_path: str,
                         idx_start: int, idx_end: int, escaped_idxs: set, queue: Queue):
    d1 = dict()
    d2 = dict()
    with open(src_file_path, mode='r', encoding='utf-8') as f1, open(tgt_file_path, mode='r', encoding='utf-8') as f2:
        for idx, (src, tgt) in enumerate(zip(f1, f2)):
            if idx < idx_start:
                continue
            if idx in escaped_idxs:
                continue
            if idx >= idx_end:
                break

            for word in src.split():
                if word in d1.keys():
                    d1[word] += 1
                else:
                    d1[word] = 1
            for word in tgt.split():
                if word in d2.keys():
                    d2[word] += 1
                else:
                    d2[word] = 1

    queue.put((d1, d2))
    return


def _corpus_numerate_worker(src_file_path: str, tgt_file_path: str,
                            idx_start: int, idx_end: int, escaped_idxs: set,
                            src_vocab: dict, tgt_vocab: dict,
                            src_sos_token: str, src_eos_token: str, src_unk_token: str,
                            tgt_sos_token: str, tgt_eos_token: str, tgt_unk_token: str,
                            queue: Queue):
    d1 = dict()
    d2 = dict()
    l1 = dict()
    l2 = dict()
    count = idx_start - sum(x in escaped_idxs for x in range(0, idx_start))
    with open(src_file_path, mode='r', encoding='utf-8') as f1, open(tgt_file_path, mode='r', encoding='utf-8') as f2:
        for idx, (src, tgt) in enumerate(zip(f1, f2)):
            if idx < idx_start:
                continue
            if idx >= idx_end:
                break
            if idx in escaped_idxs:
                continue
            src_tokens = _special_tokens_handler(src.split(), src_sos_token, src_eos_token)
            tgt_tokens = _special_tokens_handler(tgt.split(), tgt_sos_token, tgt_eos_token)
            src_enum_seq = numpy.array(list(src_vocab[word] if word in src_vocab.keys() else src_vocab[src_unk_token]
                                            for word in src_tokens)).astype(dtype=int)
            tgt_enum_seq = numpy.array(list(tgt_vocab[word] if word in tgt_vocab.keys() else tgt_vocab[tgt_unk_token]
                                            for word in tgt_tokens)).astype(dtype=int)

            d1[count] = src_enum_seq
            d2[count] = tgt_enum_seq
            l1[count] = len(src_tokens)
            l2[count] = len(tgt_tokens)

            count += 1
    queue.put((d1, d2, l1, l2))
    return


def _get_nearest_neighbors(all_lengths: torch.Tensor, pivot_lens: torch.Tensor):
    distances = all_lengths.sub(pivot_lens)
    distances_min, idx = distances.abs().sum(dim=(-1,)).topk(largest=False, k=1)

    return idx, tuple(all_lengths[idx, :].squeeze().tolist())


def _data_segment(batch_capacity: int,
                  num_of_made_segments: ValueProxy,
                  num_of_steps: int,
                  running: ValueProxy,
                  length_stats_origin: dict,
                  pool_lock: Lock,
                  batch_lock: Lock,
                  train_pool: list,
                  shuffled_times: ValueProxy,
                  logger: Stats
                  ):
    pid = os.getpid()
    print('Process %d, making segments asynchronously ... ' % pid)

    while num_of_made_segments.get() < num_of_steps:
        if not running.get():
            break

        length_stats = length_stats_origin.copy()
        all_lengths = torch.Tensor(list(length_stats.keys())).long()

        for k, v in length_stats.items():
            idxs = torch.randperm(n=len(v)).numpy()
            length_stats[k] = numpy.array(v)[idxs].tolist()

        print('Process %d, shuffling corpus done.' % pid)

        all_segments = list()

        while all_lengths.size(0) > 0:
            src_tokens = tgt_tokens = 0
            segment_temp_lengths = []
            segment_temp_idxs = []

            while src_tokens + tgt_tokens < batch_capacity:
                if src_tokens == 0:
                    src_len_max = tgt_len_max = 0
                    probs = torch.Tensor(list(len(x) for x in length_stats.values()))
                    pivot_idx = probs.multinomial(1).numpy().tolist()[0]
                    pivot_lengths = tuple(all_lengths[pivot_idx, :].tolist())
                else:
                    src_len_max = max(x[0] for x in segment_temp_lengths)
                    tgt_len_max = max(x[1] for x in segment_temp_lengths)
                    pivot_lengths = (src_len_max, tgt_len_max)
                    pivot_idx, pivot_lengths = _get_nearest_neighbors(all_lengths, torch.Tensor(pivot_lengths).long())

                src_len_max_tobe = max(src_len_max, pivot_lengths[0])
                tgt_len_max_tobe = max(tgt_len_max, pivot_lengths[1])

                examples_tochoose = length_stats[pivot_lengths]
                num_examples_tochoose = len(examples_tochoose)
                num_examples_added = len(segment_temp_idxs)
                sum_len_max_tobe = src_len_max_tobe + tgt_len_max_tobe
                src_tokens_tobe = src_len_max_tobe * (num_examples_added + num_examples_tochoose)
                tgt_tokens_tobe = tgt_len_max_tobe * (num_examples_added + num_examples_tochoose)

                capacity_tobe = src_tokens_tobe + tgt_tokens_tobe

                if capacity_tobe <= batch_capacity:
                    src_tokens = src_tokens_tobe
                    tgt_tokens = tgt_tokens_tobe
                    segment_temp_lengths += [pivot_lengths]
                    segment_temp_idxs += length_stats.pop(pivot_lengths)
                    all_lengths = torch.cat((all_lengths[:pivot_idx, :], all_lengths[pivot_idx + 1:, :]), dim=0)

                    if all_lengths.size(0) == 0:
                        break
                else:
                    num_examples_toadd = (batch_capacity - sum_len_max_tobe * num_examples_added) // sum_len_max_tobe

                    if num_examples_toadd <= 0:
                        break

                    examples_tobe = examples_tochoose[:num_examples_toadd]
                    segment_temp_idxs += examples_tobe
                    length_stats[pivot_lengths] = examples_tochoose[num_examples_toadd:]

                    break

            all_segments.append(segment_temp_idxs)

        with pool_lock:
            with batch_lock:
                idxs = torch.randperm(n=len(all_segments)).numpy()
                all_segments = numpy.array(all_segments)[idxs].tolist()
                train_pool.extend(all_segments)
                shuffled_times.set(shuffled_times.get() + 1)
                num_of_made_segments.set(num_of_made_segments.get() + len(all_segments))
                output_string = 'Process %d, making %d segments done, time: %d, total: %d' \
                                % (pid, len(all_segments), shuffled_times.get(), num_of_made_segments.get())
                print(output_string)
                logger.log_to_file(output_string)

    return


def _batch_prefetch(src_pad_idx: int,
                    tgt_pad_idx: int,
                    running: ValueProxy,
                    num_of_made_batches: ValueProxy,
                    num_of_trained_batches: ValueProxy,
                    num_of_steps: int,
                    train_pool: list,
                    train_buffer_size: int,
                    train_prefetch_size: int,
                    pool_lock: Lock,
                    batch_lock: Lock,
                    corpus_source_train_numerate: dict,
                    corpus_target_train_numerate: dict,
                    corpus_source_train_length: dict,
                    corpus_target_train_length: dict,
                    batches_source_train: Queue,
                    batches_target_train: Queue,
                    batches_source_train_length: Queue,
                    batches_target_train_length: Queue,
                    logger: Stats
                    ):
    pid = os.getpid()

    while running.get():
        if len(train_pool) < train_buffer_size + num_of_made_batches.get():
            time.sleep(1)
            continue
        elif num_of_made_batches.get() - num_of_trained_batches.get() > train_prefetch_size:
            time.sleep(1)
            continue
        elif num_of_made_batches.get() >= num_of_steps:
            time.sleep(1)
            continue
        else:
            segments = list()
            with pool_lock:
                for idx in range(num_of_made_batches.get(), train_buffer_size + num_of_made_batches.get()):
                    segments.append(train_pool[idx])

        idx_start = num_of_made_batches.get()

        src_train = list()
        tgt_train = list()
        src_train_lens = list()
        tgt_train_lens = list()

        for idx, seg in enumerate(segments, start=idx_start):
            src_train.append(pad_sequence(list(torch.from_numpy(corpus_source_train_numerate[x]) for x in seg),
                                          batch_first=True, padding_value=src_pad_idx).contiguous())
            tgt_train.append(pad_sequence(list(torch.from_numpy(corpus_target_train_numerate[x]) for x in seg),
                                          batch_first=True, padding_value=tgt_pad_idx).contiguous())
            src_train_lens.append(torch.Tensor(list(corpus_source_train_length[x] for x in seg)).int())
            tgt_train_lens.append(torch.Tensor(list(corpus_target_train_length[x] for x in seg)).int())

        with batch_lock:
            batches_source_train.put(src_train)
            batches_target_train.put(tgt_train)
            batches_source_train_length.put(src_train_lens)
            batches_target_train_length.put(tgt_train_lens)
            num_of_made_batches.set(num_of_made_batches.get() + len(segments))

            output_string = 'Process %d, train batches added: %d, totally: %d' \
                            % (pid, len(segments), num_of_made_batches.get())
            print(output_string)
            logger.log_to_file(output_string)

    return
