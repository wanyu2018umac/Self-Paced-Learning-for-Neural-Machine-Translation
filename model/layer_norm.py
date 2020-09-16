import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, emb_size: int, eps: float):
        super(LayerNorm, self).__init__()
        self.emb_size = emb_size
        self.eps = eps

        weight = nn.Parameter(torch.ones((self.emb_size, )), requires_grad=True)
        bias = nn.Parameter(torch.zeros((self.emb_size, )), requires_grad=True)

        self.register_parameter('weight', weight)
        self.register_parameter('bias', bias)

        return

    def forward(self, layer_input):
        mean = layer_input.mean(dim=-1, keepdim=True)
        std = layer_input.std(dim=-1, keepdim=True)
        return self.weight * (layer_input - mean) / (std + self.eps) + self.bias

    def train(self, mode=True):
        self.training = mode
        for child in self.children():
            child.train(mode)
        return

    def eval(self):
        self.train(False)
        return


class UnlearnableLayerNorm(nn.Module):
    def __init__(self, emb_size: int, eps: float):
        super(UnlearnableLayerNorm, self).__init__()
        self.emb_size = emb_size
        self.eps = eps

        return

    def forward(self, layer_input):
        mean = layer_input.mean(dim=-1, keepdim=True)
        std = layer_input.std(dim=-1, keepdim=True)
        return (layer_input - mean) / (std + self.eps)

    def train(self, mode=True):
        self.training = mode
        for child in self.children():
            child.train(mode)
        return

    def eval(self):
        self.train(False)
        return


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, layer_input):
        return layer_input

    def train(self, mode=True):
        self.training = mode
        for child in self.children():
            child.train(mode)
        return

    def eval(self):
        self.train(False)
        return
