import torch
from torch import nn
from torch.nn.functional import log_softmax, kl_div


class LabelSmoothing(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 padding_idx: int,
                 confidence: float):
        super(LabelSmoothing, self).__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.confidence = confidence
        self.smoothed_value = (1 - self.confidence) / (self.vocab_size - 1)

        weight = torch.zeros((1, self.vocab_size))
        weight.fill_(self.smoothed_value)
        self.register_buffer('weight', weight)

        return

    def forward(self, input_probs, ground_truth):
        probs = log_softmax(input_probs, dim=-1).view(-1, self.vocab_size - 1)
        mask = ground_truth.eq(self.padding_idx).view(-1)

        return kl_div(probs,
                      self.weight.expand(probs.size(0), -1)
                      .scatter(dim=-1, index=ground_truth.view(-1).unsqueeze(dim=-1), value=self.confidence)
                      .masked_fill(mask=mask.unsqueeze(dim=-1), value=0.0)[:, 1:],
                      reduction='none').sum(dim=-1)
