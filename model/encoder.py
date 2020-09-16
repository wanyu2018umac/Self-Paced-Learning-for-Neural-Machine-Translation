from torch import nn

from model.multihead_attention import MultiHeadAttention1
from model.positionwiseff import PositionWiseFeedForward
from model.positional_encoding import NonePositionalEncoding, StaticPositionalEncoding, LearnablePositionalEncoding
from model.layer_norm import Identity, LayerNorm, UnlearnableLayerNorm


class TransformerEncoder(nn.Module):
    def __init__(self,
                 emb_size: int, feedforward_size: int,
                 num_of_layers: int, num_of_heads: int, max_seq_length: int,
                 embedding_dropout_prob: float,
                 attention_dropout_prob: float,
                 feedforward_dropout_prob: float,
                 residual_dropout_prob: float,
                 activate_function_name: str,
                 positional_encoding: str,
                 layer_norm_pre: str,
                 layer_norm_post: str,
                 layer_norm_start: str,
                 layer_norm_end: str):
        super(TransformerEncoder, self).__init__()
        self.emb_size = emb_size
        self.feedforward_size = feedforward_size
        self.num_of_layers = num_of_layers
        self.num_of_heads = num_of_heads
        self.max_seq_length = max_seq_length + 2

        self.embedding_dropout_prob = embedding_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.feedforward_dropout_prob = feedforward_dropout_prob
        self.residual_dropout_prob = residual_dropout_prob

        self.activate_function_name = activate_function_name

        self.positional_encoding = positional_encoding

        self.layer_norm_pre = layer_norm_pre
        self.layer_norm_post = layer_norm_post
        self.layer_norm_start = layer_norm_start
        self.layer_norm_end = layer_norm_end

        self.layers = nn.ModuleList(
            list(TransformerEncoderLayer(emb_size=self.emb_size,
                                         feedforward_size=self.feedforward_size,
                                         num_of_heads=self.num_of_heads,
                                         attention_dropout_prob=self.attention_dropout_prob,
                                         residual_dropout_prob=self.residual_dropout_prob,
                                         feedforward_dropout_prob=self.feedforward_dropout_prob,
                                         activate_function_name=self.activate_function_name,
                                         layer_norm_pre=self.layer_norm_pre,
                                         layer_norm_post=self.layer_norm_post)
                 for _ in range(0, self.num_of_layers)))

        if self.positional_encoding == 'none':
            self.positional_encoding = NonePositionalEncoding(max_seq_length=self.max_seq_length,
                                                              emb_size=self.emb_size,
                                                              dropout_prob=self.embedding_dropout_prob)
        elif self.positional_encoding == 'static':
            self.positional_encoding = StaticPositionalEncoding(max_seq_length=self.max_seq_length,
                                                                emb_size=self.emb_size,
                                                                dropout_prob=self.embedding_dropout_prob)
        else:
            self.positional_encoding = LearnablePositionalEncoding(max_seq_length=self.max_seq_length,
                                                                   emb_size=self.emb_size,
                                                                   dropout_prob=self.embedding_dropout_prob)

        if self.layer_norm_start == 'learnable':
            self.encoder_layer_norm_pre = LayerNorm(emb_size=self.emb_size, eps=1e-6)
        elif self.layer_norm_start == 'static':
            self.encoder_layer_norm_pre = UnlearnableLayerNorm(emb_size=self.emb_size, eps=1e-6)
        else:
            self.encoder_layer_norm_pre = Identity()

        if self.layer_norm_end == 'learnable':
            self.encoder_layer_norm_post = LayerNorm(emb_size=self.emb_size, eps=1e-6)
        elif self.layer_norm_end == 'static':
            self.encoder_layer_norm_post = UnlearnableLayerNorm(emb_size=self.emb_size, eps=1e-6)
        else:
            self.encoder_layer_norm_post = Identity()

        return

    def init_parameters(self):
        self.positional_encoding.init_parameters()
        for i in range(0, self.num_of_layers):
            self.layers[i].init_parameters()

        return

    def forward(self, src_embs, src_mask):
        src_embs_pe = self.positional_encoding(src_embs, src_mask)
        next_input = self.encoder_layer_norm_pre(src_embs_pe)

        for i in range(0, self.num_of_layers):
            next_input = self.layers[i](next_input, src_mask)

        return self.encoder_layer_norm_post(next_input)

    def train(self, mode=True):
        self.training = mode
        for child in self.children():
            child.train(mode)
        return

    def eval(self):
        self.train(False)
        return


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 emb_size: int, feedforward_size: int, num_of_heads: int,
                 attention_dropout_prob: float,
                 residual_dropout_prob: float,
                 feedforward_dropout_prob: float,
                 activate_function_name: str,
                 layer_norm_pre: str,
                 layer_norm_post: str):
        super(TransformerEncoderLayer, self).__init__()

        self.emb_size = emb_size
        self.feedforward_size = feedforward_size
        self.num_of_heads = num_of_heads

        self.attention_dropout_prob = attention_dropout_prob
        self.feedforward_dropout_prob = feedforward_dropout_prob
        self.residual_dropout_prob = residual_dropout_prob

        self.activate_function_name = activate_function_name

        self.layer_norm_pre = layer_norm_pre
        self.layer_norm_post = layer_norm_post

        self.multihead_attention = MultiHeadAttention1(emb_size=self.emb_size,
                                                       num_of_heads=self.num_of_heads,
                                                       attention_dropout_prob=self.attention_dropout_prob,
                                                       residual_dropout_prob=self.residual_dropout_prob,
                                                       layer_norm_pre=self.layer_norm_pre,
                                                       layer_norm_post=self.layer_norm_post)
        self.pwff = PositionWiseFeedForward(emb_size=self.emb_size,
                                            feedforward_size=self.feedforward_size,
                                            feedforward_dropout_prob=self.feedforward_dropout_prob,
                                            residual_dropout_prob=self.residual_dropout_prob,
                                            activate_function_name=self.activate_function_name,
                                            layer_norm_pre=self.layer_norm_pre,
                                            layer_norm_post=self.layer_norm_post)
        return

    def init_parameters(self):
        self.multihead_attention.init_parameters()
        self.pwff.init_parameters()

        return

    def forward(self, src_input, src_mask):
        output1 = self.multihead_attention(src_input, src_mask)
        output2 = self.pwff(output1)

        return output2

    def train(self, mode=True):
        self.training = mode
        for child in self.children():
            child.train(mode)
        return

    def eval(self):
        self.train(False)
        return
