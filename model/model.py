import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from model.embeddings_and_logits import IndependentEmbeddingsAndLogits, IndependentEmbeddingsSharedLogits, \
    SharedEmbeddingsIndependentLogits, SharedEmbeddingsAndLogits
from model.encoder import TransformerEncoder
from model.decoder import TransformerDecoder


class TransformerMT(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 joint_vocab_size: int,
                 share_embedding: bool,
                 share_projection_and_embedding: bool,
                 src_pad_idx: int,
                 tgt_pad_idx: int,
                 tgt_sos_idx: int,
                 tgt_eos_idx: int,
                 positional_encoding: str,
                 emb_size: int,
                 feed_forward_size: int,
                 num_of_layers: int,
                 num_of_heads: int,
                 train_max_seq_length: int,
                 infer_max_seq_length: int,
                 infer_max_seq_length_mode: str,
                 batch_size: int,
                 update_decay: int,
                 embedding_dropout_prob: float,
                 attention_dropout_prob: float,
                 feedforward_dropout_prob: float,
                 residual_dropout_prob: float,
                 activate_function_name: str,
                 emb_norm_clip: float,
                 emb_norm_clip_type: float,
                 layer_norm_pre: str,
                 layer_norm_post: str,
                 layer_norm_encoder_start: str,
                 layer_norm_encoder_end: str,
                 layer_norm_decoder_start: str,
                 layer_norm_decoder_end: str,
                 prefix: str,
                 pretrained_src_emb: str,
                 pretrained_tgt_emb: str,
                 pretrained_src_eos: str,
                 pretrained_tgt_eos: str,
                 src_vocab: dict,
                 tgt_vocab: dict,
                 criterion: nn.Module):
        super(TransformerMT, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.joint_vocab_size = joint_vocab_size

        self.share_embedding = share_embedding
        self.share_projection_and_embedding = share_projection_and_embedding

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_sos_idx = tgt_sos_idx
        self.tgt_eos_idx = tgt_eos_idx

        self.positional_encoding = positional_encoding

        self.emb_size = emb_size
        self.feed_forward_size = feed_forward_size
        self.num_of_layers = num_of_layers
        self.num_of_heads = num_of_heads
        self.train_max_seq_length = train_max_seq_length
        self.infer_max_seq_length = infer_max_seq_length
        self.infer_max_seq_length_mode = infer_max_seq_length_mode
        self.batch_size = batch_size

        self.embedding_dropout_prob = embedding_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.feedforward_dropout_prob = feedforward_dropout_prob
        self.residual_dropout_prob = residual_dropout_prob

        self.activate_function_name = activate_function_name

        self.emb_norm_clip = emb_norm_clip
        self.emb_norm_clip_type = emb_norm_clip_type

        self.layer_norm_pre = layer_norm_pre
        self.layer_norm_post = layer_norm_post
        self.layer_norm_encoder_start = layer_norm_encoder_start
        self.layer_norm_encoder_end = layer_norm_encoder_end
        self.layer_norm_decoder_start = layer_norm_decoder_start
        self.layer_norm_decoder_end = layer_norm_decoder_end

        self.criterion = criterion

        if self.share_embedding and self.share_projection_and_embedding:
            self.embeddings = SharedEmbeddingsAndLogits(vocab_size=self.joint_vocab_size,
                                                        embedding_size=self.emb_size,
                                                        pad_idx=self.tgt_pad_idx,
                                                        max_norm=self.emb_norm_clip,
                                                        max_norm_type=self.emb_norm_clip_type)
        elif self.share_embedding:
            self.embeddings = SharedEmbeddingsIndependentLogits(vocab_size=self.joint_vocab_size,
                                                                embedding_size=self.emb_size,
                                                                pad_idx=self.tgt_pad_idx,
                                                                max_norm=self.emb_norm_clip,
                                                                max_norm_type=self.emb_norm_clip_type)
        elif self.share_projection_and_embedding:
            self.embeddings = IndependentEmbeddingsSharedLogits(src_vocab_size=self.src_vocab_size,
                                                                tgt_vocab_size=self.tgt_vocab_size,
                                                                embedding_size=self.emb_size,
                                                                src_pad_idx=self.src_pad_idx,
                                                                tgt_pad_idx=self.tgt_pad_idx,
                                                                max_norm=self.emb_norm_clip,
                                                                max_norm_type=self.emb_norm_clip_type)
        else:
            self.embeddings = IndependentEmbeddingsAndLogits(src_vocab_size=self.src_vocab_size,
                                                             tgt_vocab_size=self.tgt_vocab_size,
                                                             embedding_size=self.emb_size,
                                                             src_pad_idx=self.src_pad_idx,
                                                             tgt_pad_idx=self.tgt_pad_idx,
                                                             max_norm=self.emb_norm_clip,
                                                             max_norm_type=self.emb_norm_clip_type)

        self.encoder = TransformerEncoder(emb_size=self.emb_size,
                                          feedforward_size=self.feed_forward_size,
                                          num_of_layers=self.num_of_layers,
                                          num_of_heads=self.num_of_heads,
                                          max_seq_length=max(self.train_max_seq_length, self.infer_max_seq_length),
                                          embedding_dropout_prob=self.embedding_dropout_prob,
                                          attention_dropout_prob=self.attention_dropout_prob,
                                          positional_encoding=self.positional_encoding,
                                          residual_dropout_prob=self.residual_dropout_prob,
                                          feedforward_dropout_prob=self.feedforward_dropout_prob,
                                          activate_function_name=self.activate_function_name,
                                          layer_norm_pre=self.layer_norm_pre,
                                          layer_norm_post=self.layer_norm_post,
                                          layer_norm_start=self.layer_norm_encoder_start,
                                          layer_norm_end=self.layer_norm_encoder_end)
        self.decoder = TransformerDecoder(emb_size=self.emb_size,
                                          feedforward_size=self.feed_forward_size,
                                          num_of_layers=self.num_of_layers,
                                          num_of_heads=self.num_of_heads,
                                          max_seq_length=max(self.train_max_seq_length, self.infer_max_seq_length),
                                          embedding_dropout_prob=self.embedding_dropout_prob,
                                          attention_dropout_prob=self.attention_dropout_prob,
                                          residual_dropout_prob=self.residual_dropout_prob,
                                          feedforward_dropout_prob=self.feedforward_dropout_prob,
                                          activate_function_name=self.activate_function_name,
                                          positional_encoding=self.positional_encoding,
                                          layer_norm_pre=self.layer_norm_pre,
                                          layer_norm_post=self.layer_norm_post,
                                          layer_norm_start=self.layer_norm_decoder_start,
                                          layer_norm_end=self.layer_norm_decoder_end)

        if pretrained_src_emb != '':
            print('Load pretrained source embeddings from', pretrained_src_emb)
            with open(prefix + pretrained_src_emb, 'r') as f:
                for _, line in enumerate(f):
                    splits = line.split()
                    if len(splits) <= self.emb_size:
                        continue
                    if splits[0] == pretrained_src_eos:
                        splits[0] = '<EOS>'

                    if splits[0] in src_vocab.keys():
                        self.encoder.src_embs.weight.data.index_copy_(
                            dim=0,
                            index=torch.Tensor([src_vocab[splits[0]]]).long(),
                            source=torch.Tensor(list(float(x) for x in splits[1:])).unsqueeze(dim=0))

        if pretrained_tgt_emb != '':
            print('Load pretrained target embeddings from', pretrained_tgt_emb)
            with open(prefix + pretrained_tgt_emb, 'r') as f:
                for _, line in enumerate(f):
                    splits = line.split()
                    if len(splits) <= self.emb_size:
                        continue
                    if splits[0] == pretrained_tgt_eos:
                        splits[0] = '<EOS>'

                    if splits[0] in tgt_vocab.keys():
                        self.decoder.tgt_embs.weight.data.index_copy_(
                            dim=0,
                            index=torch.Tensor([tgt_vocab[splits[0]]]).long(),
                            source=torch.Tensor(list(float(x) for x in splits[1:])).unsqueeze(dim=0))

        self.softmax = torch.softmax

        triu_backup = torch.ones((1024, 1024)).triu(1).eq(1)
        mask_backup = torch.arange(0, 1024, requires_grad=False).unsqueeze(dim=0).int()

        self.register_buffer(name='triu_backup', tensor=triu_backup)
        self.register_buffer(name='mask_backup', tensor=mask_backup)

        self.update_decay = update_decay

        return

    def init_parameters(self):
        with torch.no_grad():
            self.embeddings.init_parameters()
            self.encoder.init_parameters()
            self.decoder.init_parameters()
        return

    def forward(self, source_enumerate, target_enumerate, src_mask, tgt_mask, crs_mask, ground_truth):
        src_embs = self.embeddings.get_src_embs(source_enumerate)
        # context_vector = checkpoint(self.encoder, src_embs, src_mask)
        context_vector = self.encoder(src_embs, src_mask)

        tgt_embs = self.embeddings.get_tgt_embs(target_enumerate)
        # output = checkpoint(self.decoder, tgt_embs, tgt_mask, context_vector, crs_mask)
        output = self.decoder(tgt_embs, tgt_mask, context_vector, crs_mask)
        logits = self.embeddings.get_logits(output)

        return torch.argmax(logits, dim=-1) + 1, self.criterion(logits, ground_truth)

    def model_parameters_statistic(self):
        logs = ['Parameters: %d' % sum(p.numel() for p in self.parameters())]

        for name, parameters in self.named_parameters():
            logs.append('%8d\t%20s\t%s' % (parameters.numel(), list(parameters.size()), name))

        logs.append('*' * 80)

        print('\n'.join(logs))
        return '\n'.join(logs)

    def train(self, mode=True):
        self.training = mode
        for child in self.children():
            child.train(mode)
        return

    def eval(self):
        self.train(False)
        return
