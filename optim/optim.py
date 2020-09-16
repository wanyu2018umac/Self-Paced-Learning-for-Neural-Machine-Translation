from torch.optim.optimizer import Optimizer


def set_lr(optim: Optimizer, lr: float):
    for group in optim.param_groups:
        group['lr'] = lr
    return
