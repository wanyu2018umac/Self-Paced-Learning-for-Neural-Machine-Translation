import torch
from torch import nn

import math


class StaticPositionalEncoding(nn.Module):
    def __init__(self,
                 max_seq_length: int,
                 emb_size: int,
                 dropout_prob: float):
        super(StaticPositionalEncoding, self).__init__()
        self.max_seq_length = max_seq_length
        self.emb_size = emb_size
        self.dropout_prob = dropout_prob

        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.emb_multiply_factor = math.sqrt(self.emb_size)

        position_encoding = torch.zeros([1024, self.emb_size])
        position = torch.arange(0, 1024).float().unsqueeze(dim=1)
        div_term = torch.exp(torch.arange(0, self.emb_size, 2, dtype=torch.float) *
                             (-(math.log(10000.0) / self.emb_size)))

        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        position_encoding = position_encoding.unsqueeze(dim=0).requires_grad_(False)
        self.register_buffer('position_encoding', position_encoding)

        return

    def init_parameters(self):
        return

    def forward(self, input_embs, input_mask):
        input_embs = input_embs * self.emb_multiply_factor
        positional_embs = input_embs + self.position_encoding[:, :input_embs.size(1), :]
        return self.dropout(positional_embs)

    def train(self, mode=True):
        self.training = mode
        for child in self.children():
            child.train(mode)
        return

    def eval(self):
        self.train(False)
        return


class LearnablePositionalEncoding(nn.Module):
    def __init__(self,
                 max_seq_length: int,
                 emb_size: int,
                 dropout_prob: float):
        super(LearnablePositionalEncoding, self).__init__()
        self.max_seq_length = max_seq_length
        self.emb_size = emb_size
        self.dropout_prob = dropout_prob

        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.emb_multiply_factor = math.sqrt(self.emb_size)
        self.positional_encoding = nn.Parameter(data=torch.zeros([1, 1024, self.emb_size]),
                                                requires_grad=True)

        return

    def init_parameters(self):
        bound = math.sqrt(3 / self.max_seq_length)
        self.positional_encoding.uniform_(-bound, bound)
        return

    def forward(self, input_embs, input_mask):
        input_embs = input_embs * self.emb_multiply_factor
        positional_embs = input_embs + self.positional_encoding[:, :input_embs.size(1), :]
        return self.dropout(positional_embs)

    def train(self, mode=True):
        self.training = mode
        for child in self.children():
            child.train(mode)
        return

    def eval(self):
        self.train(False)
        return


class NonePositionalEncoding(nn.Module):
    def __init__(self,
                 max_seq_length: int,
                 emb_size: int,
                 dropout_prob: float):
        super(NonePositionalEncoding, self).__init__()
        self.max_seq_length = max_seq_length
        self.emb_size = emb_size
        self.dropout_prob = dropout_prob

        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.emb_multiply_factor = math.sqrt(self.emb_size)
        return

    def init_parameters(self):
        return

    def forward(self, input_embs, input_mask):
        input_embs = input_embs * self.emb_multiply_factor
        return self.dropout(input_embs)

    def train(self, mode=True):
        self.training = mode
        for child in self.children():
            child.train(mode)
        return

    def eval(self):
        self.train(False)
        return
