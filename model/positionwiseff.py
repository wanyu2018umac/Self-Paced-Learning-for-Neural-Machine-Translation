import torch
from torch import nn
import torch.nn.functional as functional

from model.layer_norm import Identity, LayerNorm, UnlearnableLayerNorm


class PositionWiseFeedForward(nn.Module):
    def __init__(self, emb_size: int, feedforward_size: int,
                 feedforward_dropout_prob: float,
                 residual_dropout_prob: float,
                 activate_function_name: str,
                 layer_norm_pre: str,
                 layer_norm_post: str):
        super(PositionWiseFeedForward, self).__init__()
        self.emb_size = emb_size
        self.feedforward_size = feedforward_size
        self.feedforward_dropout_prob = feedforward_dropout_prob
        self.residual_dropout_prob = residual_dropout_prob

        self.activate_function_name = activate_function_name

        self.layer_norm_pre = layer_norm_pre
        self.layer_norm_post = layer_norm_post

        self.linear_in_weight = nn.Parameter(torch.zeros(size=(self.emb_size, self.feedforward_size)), requires_grad=True)
        self.linear_in_bias = nn.Parameter(torch.zeros(size=(self.feedforward_size, )), requires_grad=True)

        self.linear_out_weight = nn.Parameter(torch.zeros(size=(self.feedforward_size, self.emb_size)), requires_grad=True)
        self.linear_out_bias = nn.Parameter(torch.zeros(size=(self.emb_size, )), requires_grad=True)

        if self.activate_function_name == 'softplus':
            self.activate_function = functional.softplus
        elif self.activate_function_name == 'gelu':
            self.activate_function = functional.gelu
        else:
            self.activate_function = functional.relu

        self.dropout_feedforward = nn.Dropout(p=self.feedforward_dropout_prob, inplace=True)
        self.dropout_residual = nn.Dropout(p=self.residual_dropout_prob, inplace=True)

        if self.layer_norm_pre == 'learnable':
            self.layer_norm_pwff_pre = LayerNorm(emb_size=self.emb_size, eps=1e-6)
        elif self.layer_norm_pre == 'static':
            self.layer_norm_pwff_pre = UnlearnableLayerNorm(emb_size=self.emb_size, eps=1e-6)
        else:
            self.layer_norm_pwff_pre = Identity()

        if self.layer_norm_post == 'learnable':
            self.layer_norm_pwff_post = LayerNorm(emb_size=self.emb_size, eps=1e-6)
        elif self.layer_norm_post == 'static':
            self.layer_norm_pwff_post = UnlearnableLayerNorm(emb_size=self.emb_size, eps=1e-6)
        else:
            self.layer_norm_pwff_post = Identity()
        return

    def init_parameters(self):
        bound = (6 / (self.feedforward_size + self.emb_size)) ** 0.5
        self.linear_in_weight.data.uniform_(-bound, bound)
        self.linear_out_weight.data.uniform_(-bound, bound)

        bound = (3 / self.feedforward_size) ** 0.5
        self.linear_in_bias.data.uniform_(-bound, bound)

        bound = (3 / self.emb_size) ** 0.5
        self.linear_out_bias.data.uniform_(-bound, bound)

        return

    def forward(self, attention_vector):
        normalized_input = self.layer_norm_pwff_pre(attention_vector)
        input_embs = torch.add(torch.matmul(normalized_input, self.linear_in_weight), self.linear_in_bias)
        trans_embs = self.dropout_feedforward(self.activate_function(input_embs))
        output_embs = self.dropout_residual(torch.add(torch.matmul(trans_embs, self.linear_out_weight), self.linear_out_bias))
        residual = output_embs + attention_vector
        normalized = self.layer_norm_pwff_post(residual)
        return normalized

    def train(self, mode=True):
        self.training = mode
        for child in self.children():
            child.train(mode)
        return

    def eval(self):
        self.train(False)
        return
