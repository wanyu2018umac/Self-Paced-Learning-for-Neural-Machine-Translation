import torch
from torch import nn
from torch.nn.functional import softmax, nll_loss


class CrossEntropy(nn.Module):
    def __init__(self, token_weight: torch.Tensor):
        super(CrossEntropy, self).__init__()
        self.token_weight = token_weight

        return

    def forward(self, output_probs, classification):
        softmax_probs = softmax(output_probs, dim=-1)
        nll = nll_loss(softmax_probs, classification, weight=self.token_weight)
        return nll
