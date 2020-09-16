import torch
from torch.autograd.function import Function


def split_heads(input_tensor: torch.Tensor, batch_size: int, length_out: int, num_of_heads: int):
    return input_tensor.view(batch_size, length_out, num_of_heads, -1).contiguous().transpose(1, 2).contiguous()


def attention_alignment(query: torch.Tensor, key: torch.Tensor):
    return query.matmul(key.transpose(-2, -1).contiguous())


def masked_alignment(alignment: torch.Tensor, input_mask: torch.Tensor, masked_value: float):
    mask = input_mask.unsqueeze(dim=1).expand_as(alignment)
    return alignment.masked_fill(mask=mask, value=masked_value)


def softmax(input_tensor: torch.Tensor):
    probs = input_tensor.exp()
    return probs.div(probs.sum(dim=(-1, ), keepdim=True))


def logsoftmax(input_tensor: torch.Tensor):
    return softmax(input_tensor).add(1e-20).log()


def cross_entropy(predicted: torch.Tensor, true: torch.Tensor):
    return predicted.mul(true.add(1e-20).log().neg()).mean(dim=(-1, ))


def entropy(probability: torch.Tensor):
    return probability.mul(probability.add(1e-20).log().neg()).sum(dim=(-1, ))


def kl_divergence(predicted: torch.Tensor, true: torch.Tensor):
    return cross_entropy(predicted, true).sub(entropy(true))


def chunk(src: torch.Tensor, tgt: torch.Tensor, src_lens: torch.Tensor, tgt_lens: torch.Tensor, cl_factors: torch.Tensor, num_of_chunks: int):
    return list(zip(src.chunk(chunks=num_of_chunks, dim=0),
                    tgt.chunk(chunks=num_of_chunks, dim=0),
                    src_lens.chunk(chunks=num_of_chunks, dim=0),
                    tgt_lens.chunk(chunks=num_of_chunks, dim=0),
                    cl_factors.chunk(chunks=num_of_chunks, dim=0)))


def shifted_target(target_tensor: torch.Tensor, tgt_eos_idx: int, tgt_pad_idx: int):
    target_input = target_tensor[:, :-1]
    target_input = target_input.masked_fill(target_input.eq(tgt_eos_idx), tgt_pad_idx)
    target_output = target_tensor[:, 1:]

    return target_input.contiguous(), target_output.contiguous()


def generate_train_mask(mask_backup: torch.Tensor,
                        triu_backup: torch.Tensor,
                        source_lengths: torch.Tensor,
                        target_lengths: torch.Tensor,
                        batch_size: int,
                        src_len_max: int,
                        tgt_len_max: int):
    src_mask = mask_backup[:, :src_len_max].repeat(batch_size, 1).ge(source_lengths.unsqueeze(dim=1)). \
        unsqueeze(dim=1).repeat(1, src_len_max, 1)
    tgt_mask = triu_backup[:tgt_len_max, :tgt_len_max].unsqueeze(dim=0).repeat(batch_size, 1, 1) | \
               mask_backup[:, :tgt_len_max].repeat(batch_size, 1).ge(target_lengths.unsqueeze(dim=1)). \
                   unsqueeze(dim=1).repeat(1, tgt_len_max, 1)
    crs_mask = mask_backup[:, :src_len_max].repeat(batch_size, 1).ge(source_lengths.unsqueeze(dim=1)). \
        unsqueeze(dim=1).repeat(1, tgt_len_max, 1)

    return src_mask, tgt_mask, crs_mask


def generate_loss_mask(mask_backup: torch.Tensor,
                       batch_size: int,
                       target_lengths: torch.Tensor,
                       tgt_len_max: int,
                       backward_factor: float,
                       backward_cl_factor: torch.Tensor):
    target_lengths = target_lengths.unsqueeze(dim=-1)
    loss_mask = mask_backup[:, :tgt_len_max].repeat(batch_size, 1).lt(target_lengths).float(). \
        mul(backward_factor).div(target_lengths.float())

    return (loss_mask * backward_cl_factor).view(-1)
