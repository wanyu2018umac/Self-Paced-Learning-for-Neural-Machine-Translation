import torch
from torch import nn
from torch.nn.functional import embedding


class IndependentEmbeddingsAndLogits(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 embedding_size: int,
                 src_pad_idx: int,
                 tgt_pad_idx: int,
                 max_norm: float,
                 max_norm_type: float):
        super(IndependentEmbeddingsAndLogits, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding_size = embedding_size
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.max_norm = max_norm if max_norm > 0 else None
        self.max_norm_type = max_norm_type

        self.src_embs = nn.Parameter(torch.zeros(size=(self.src_vocab_size, self.embedding_size), requires_grad=True))
        self.tgt_embs = nn.Parameter(torch.zeros(size=(self.tgt_vocab_size, self.embedding_size), requires_grad=True))
        self.logits = nn.Parameter(torch.zeros(size=(self.embedding_size, self.tgt_vocab_size - 1), requires_grad=True))
        return

    def init_parameters(self):
        var = self.embedding_size ** -0.5
        self.src_embs.data.normal_(0, var)
        self.tgt_embs.data.normal_(0, var)
        self.src_embs.data[self.src_pad_idx, :] = 0.0
        self.tgt_embs.data[self.tgt_pad_idx, :] = 0.0
        self.logits.data.normal_(0, var)

        return

    def get_src_embs(self, source_enumerate: torch.Tensor):
        return embedding(input=source_enumerate,
                         weight=self.src_embs,
                         padding_idx=self.src_pad_idx,
                         max_norm=self.max_norm,
                         norm_type=self.max_norm_type,
                         scale_grad_by_freq=False,
                         sparse=False)

    def get_tgt_embs(self, target_enumerate: torch.Tensor):
        return embedding(input=target_enumerate,
                         weight=self.tgt_embs,
                         padding_idx=self.tgt_pad_idx,
                         max_norm=self.max_norm,
                         norm_type=self.max_norm_type,
                         scale_grad_by_freq=False,
                         sparse=False)

    def get_logits(self, embeddings: torch.Tensor):
        return torch.matmul(embeddings, self.logits)

    def zero_pad_emb(self):
        self.src_embs.data[self.src_pad_idx, :].zero_()
        self.tgt_embs.data[self.tgt_pad_idx, :].zero_()
        return

    def forward(self):
        return None

    def train(self, mode=True):
        self.training = mode
        for child in self.children():
            child.train(mode)
        return

    def eval(self):
        self.train(False)
        return


class IndependentEmbeddingsSharedLogits(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 embedding_size: int,
                 src_pad_idx: int,
                 tgt_pad_idx: int,
                 max_norm: float,
                 max_norm_type: float):
        super(IndependentEmbeddingsSharedLogits, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding_size = embedding_size
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.max_norm = max_norm if max_norm > 0 else None
        self.max_norm_type = max_norm_type

        self.src_embs = nn.Parameter(torch.zeros(size=(self.src_vocab_size, self.embedding_size), requires_grad=True))
        self.tgt_embs = nn.Parameter(torch.zeros(size=(self.tgt_vocab_size, self.embedding_size), requires_grad=True))
        return

    def init_parameters(self):
        var = self.embedding_size ** -0.5
        self.src_embs.data.normal_(0, var)
        self.tgt_embs.data.normal_(0, var)
        self.src_embs.data[self.src_pad_idx, :] = 0.0
        self.tgt_embs.data[self.tgt_pad_idx, :] = 0.0

        return

    def get_src_embs(self, source_enumerate: torch.Tensor):
        return embedding(input=source_enumerate,
                         weight=self.src_embs,
                         padding_idx=self.src_pad_idx,
                         max_norm=self.max_norm,
                         norm_type=self.max_norm_type,
                         scale_grad_by_freq=False,
                         sparse=False)

    def get_tgt_embs(self, target_enumerate: torch.Tensor):
        return embedding(input=target_enumerate,
                         weight=self.tgt_embs,
                         padding_idx=self.tgt_pad_idx,
                         max_norm=self.max_norm,
                         norm_type=self.max_norm_type,
                         scale_grad_by_freq=False,
                         sparse=False)

    def get_logits(self, embeddings: torch.Tensor):
        return torch.matmul(embeddings, self.tgt_embs.t()[:, 1:])

    def zero_pad_emb(self):
        self.src_embs.data[self.src_pad_idx, :].zero_()
        self.tgt_embs.data[self.tgt_pad_idx, :].zero_()
        return

    def forward(self):
        return None

    def train(self, mode=True):
        self.training = mode
        for child in self.children():
            child.train(mode)
        return

    def eval(self):
        self.train(False)
        return


class SharedEmbeddingsIndependentLogits(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_size: int,
                 pad_idx: int,
                 max_norm: float,
                 max_norm_type: float):
        super(SharedEmbeddingsIndependentLogits, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.pad_idx = pad_idx
        self.max_norm = max_norm if max_norm > 0 else None
        self.max_norm_type = max_norm_type

        self.embs = nn.Parameter(torch.zeros(size=(self.vocab_size, self.embedding_size), requires_grad=True))
        self.logits = nn.Parameter(torch.zeros(size=(self.embedding_size, self.vocab_size - 1)), requires_grad=True)
        return

    def init_parameters(self):
        var = self.embedding_size ** -0.5
        self.embs.data.normal_(0, var)
        self.embs.data[self.pad_idx, :] = 0.0
        self.logits.data.normal_(0, var)

        return

    def get_src_embs(self, source_enuemrate: torch.Tensor):
        return embedding(input=source_enuemrate,
                         weight=self.embs,
                         padding_idx=self.pad_idx,
                         max_norm=self.max_norm,
                         norm_type=self.max_norm_type,
                         scale_grad_by_freq=False,
                         sparse=False)

    def get_tgt_embs(self, target_enumerate: torch.Tensor):
        return embedding(input=target_enumerate,
                         weight=self.embs,
                         padding_idx=self.pad_idx,
                         max_norm=self.max_norm,
                         norm_type=self.max_norm_type,
                         scale_grad_by_freq=False,
                         sparse=False)

    def get_logits(self, embeddings: torch.Tensor):
        return torch.matmul(embeddings, self.logits)

    def zero_pad_emb(self):
        self.embs.data[self.pad_idx, :].zero_()
        return

    def forward(self):
        return None

    def train(self, mode=True):
        self.training = mode
        for child in self.children():
            child.train(mode)
        return

    def eval(self):
        self.train(False)
        return


class SharedEmbeddingsAndLogits(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_size: int,
                 pad_idx: int,
                 max_norm: float,
                 max_norm_type: float):
        super(SharedEmbeddingsAndLogits, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.pad_idx = pad_idx
        self.max_norm = max_norm if max_norm > 0 else None
        self.max_norm_type = max_norm_type

        self.embs = nn.Parameter(torch.zeros(size=(self.vocab_size, self.embedding_size), requires_grad=True))
        return

    def init_parameters(self):
        var = self.embedding_size ** -0.5
        self.embs.data.normal_(0, var)
        self.embs.data[self.pad_idx, :] = 0.0

        return

    def get_src_embs(self, source_enuemrate: torch.Tensor):
        return embedding(input=source_enuemrate,
                         weight=self.embs,
                         padding_idx=self.pad_idx,
                         max_norm=self.max_norm,
                         norm_type=self.max_norm_type,
                         scale_grad_by_freq=False,
                         sparse=False)

    def get_tgt_embs(self, target_enumerate: torch.Tensor):
        return embedding(input=target_enumerate,
                         weight=self.embs,
                         padding_idx=self.pad_idx,
                         max_norm=self.max_norm,
                         norm_type=self.max_norm_type,
                         scale_grad_by_freq=False,
                         sparse=False)

    def get_logits(self, embeddings: torch.Tensor):
        return torch.matmul(embeddings, self.embs.t()[:, 1:])

    def zero_pad_emb(self):
        self.embs.data[self.pad_idx, :].zero_()
        return

    def forward(self):
        return None

    def train(self, mode=True):
        self.training = mode
        for child in self.children():
            child.train(mode)
        return

    def eval(self):
        self.train(False)
        return
