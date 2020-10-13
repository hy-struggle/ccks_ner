# /usr/bin/env python
# coding=utf-8
"""TENER model"""
import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativeEmbedding(nn.Module):
    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen].
        """
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + seq_len
        if max_pos > self.origin_shift:
            # recompute/expand embeddings if needed
            weights = self.get_embedding(
                max_pos * 2,
                self.embedding_dim,
                self.padding_idx,
            )
            weights = weights.to(self._float_tensor)
            del self.weights
            self.origin_shift = weights.size(0) // 2
            self.register_buffer('weights', weights)

        positions = torch.arange(-seq_len, seq_len).to(input.device).long() + self.origin_shift  # 2*seq_len
        embed = self.weights.index_select(0, positions.long()).detach()
        return embed


class RelativeSinusoidalPositionalEmbedding(RelativeEmbedding):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        """

        :param embedding_dim: 每个位置的dimension
        :param padding_idx:
        :param init_size:
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        assert init_size % 2 == 0
        weights = self.get_embedding(
            init_size + 1,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('weights', weights)
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def get_embedding(self, num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(-num_embeddings // 2, num_embeddings // 2, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        self.origin_shift = num_embeddings // 2 + 1
        return emb


class RelativeMultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_head, dropout, r_w_bias=None, r_r_bias=None, scale=False):
        """

        :param int d_model:
        :param int n_head:
        :param dropout: 对attention map的dropout
        :param r_w_bias: n_head x head_dim or None, 如果为dim
        :param r_r_bias: n_head x head_dim or None,
        :param scale:
        :param rel_pos_embed:
        """
        super().__init__()
        self.qv_linear = nn.Linear(d_model, d_model * 2, bias=False)
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout_layer = nn.Dropout(dropout)

        self.pos_embed = RelativeSinusoidalPositionalEmbedding(d_model // n_head, 0, 1200)

        if scale:
            self.scale = math.sqrt(d_model // n_head)
        else:
            self.scale = 1

        if r_r_bias is None or r_w_bias is None:  # Biases are not shared
            self.r_r_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, d_model // n_head)))
            self.r_w_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, d_model // n_head)))
        else:
            self.r_r_bias = r_r_bias  # r_r_bias就是v
            self.r_w_bias = r_w_bias  # r_w_bias就是u

    def forward(self, x, mask):
        """

        :param x: batch_size x max_len x d_model
        :param mask: batch_size x max_len
        :return:
        """

        batch_size, max_len, d_model = x.size()
        pos_embed = self.pos_embed(mask)  # l x head_dim

        qv = self.qv_linear(x)  # batch_size x max_len x d_model2
        q, v = torch.chunk(qv, chunks=2, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = x.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)  # b x n x l x d

        rw_head_q = q + self.r_r_bias[:, None]
        AC = torch.einsum('bnqd,bnkd->bnqk', [rw_head_q, k])  # b x n x l x d, n是head

        D_ = torch.einsum('nd,ld->nl', self.r_w_bias, pos_embed)[None, :, None]  # head x 2max_len, 每个head对位置的bias
        B_ = torch.einsum('bnqd,ld->bnql', q, pos_embed)  # bsz x head  x max_len x 2max_len，每个query对每个shift的偏移
        BD = B_ + D_  # bsz x head x max_len x 2max_len, 要转换为bsz x head x max_len x max_len
        BD = self._shift(BD)
        attn = AC + BD

        attn = attn / self.scale

        attn = attn.masked_fill(mask[:, None, None, :].eq(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, max_len, d_model)  # b x n x l x d

        return v

    def _shift(self, BD):
        """
        类似
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2

        转换为
        0   1  2
        -1  0  1
        -2 -1  0

        :param BD: batch_size x n_head x max_len x 2max_len
        :return: batch_size x n_head x max_len x max_len
        """
        bsz, n_head, max_len, _ = BD.size()
        zero_pad = BD.new_zeros(bsz, n_head, max_len, 1)
        BD = torch.cat([BD, zero_pad], dim=-1).view(bsz, n_head, -1, max_len)  # bsz x n_head x (2max_len+1) x max_len
        BD = BD[:, :, :-1].view(bsz, n_head, max_len, -1)  # bsz x n_head x 2max_len x max_len
        BD = BD[:, :, :, max_len:]
        return BD


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
                   torch.cumsum(mask, dim=1).type_as(mask) * mask
           ).long() + padding_idx


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        positions = make_positions(input, self.padding_idx)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)

    def forward(self, input):
        # positions: batch_size x max_len, 把words的index输入就好了
        positions = make_positions(input, self.padding_idx)
        return super().forward(positions)


class TransformerLayer(nn.Module):
    """relative self-attention layer"""

    def __init__(self, d_model, self_attn, dropout):
        """
        :param int d_model: 一般512之类的
        :param self_attn: self attention模块，输入为x:batch_size x max_len x d_model, mask:batch_size x max_len, 输出为
            batch_size x max_len x d_model
        :param float dropout: 一共三个位置的dropout的大小
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = self_attn

        self.ffn = nn.Sequential(nn.Linear(d_model, d_model),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 )

    def forward(self, x, mask):
        """
        :param x: batch_size x max_len x hidden_size
        :param mask: batch_size x max_len, 为0的地方为pad
        :return: batch_size x max_len x hidden_size
        """
        residual = x
        x = self.self_attn(x, mask)
        x = x + residual
        x = self.norm1(x)

        residual = x
        x = self.ffn(x)
        x = residual + x
        x = self.norm2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_head, dropout,
                 scale=False, dropout_attn=None, pos_embed=None):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.d_model = d_model

        if pos_embed is None:
            # 使用relative position(论文中的方法)
            self.pos_embed = None
        elif pos_embed == 'sin':
            # 使用sin position
            self.pos_embed = SinusoidalPositionalEmbedding(d_model, 0, init_size=1024)
        elif pos_embed == 'fix':
            # 使用fix position
            self.pos_embed = LearnedPositionalEmbedding(1024, d_model, 0)

        self_attn = RelativeMultiHeadAttn(d_model, n_head, dropout_attn, scale=scale)

        self.layers = nn.ModuleList(
            [TransformerLayer(d_model, deepcopy(self_attn), dropout)
             for _ in range(num_layers)])

    def forward(self, x, mask):
        """
        :param x: (batch_size x max_len x d_model[num_heads * head_dims])
        :param mask: (batch_size x max_len). 有value的地方为1.
        :return: x: after relative self-attention. (batch_size x max_len x d_model[num_heads * head_dims])
        """
        # char embedding + position embedding
        if self.pos_embed is not None:
            x = x + self.pos_embed(mask)

        # transformer encoder
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TENER(nn.Module):
    def __init__(self, tag_size, embed_size, dropout, num_layers=1, d_model=256, n_head=4,
                 fc_dropout=0.5, pos_embed=None, scale=False, dropout_attn=None):
        """

        :param tag_size: num of tags.
        :param embed_size: bert output hidden_size.
        :param num_layers: number of self-attention layers
        :param d_model: input size. num_heads * head_dims
        :param n_head: number of head
        :param feedforward_dim: the dimension of ffn
        :param dropout: dropout in transformer-encoder-ffn
        :param after_norm: normalization place
        :param attn_type: adatrans, naive
        :param fc_dropout: dropout rate before the fc layer
        :param pos_embed: position embedding的类型，支持sin, fix, None. relative时可为None
        :param scale: 是否有归一化因子
        :param dropout_attn: attention dropout prop.

        """
        super().__init__()

        self.in_fc = nn.Linear(embed_size, d_model)

        # ada-trans
        self.transformer = TransformerEncoder(num_layers, d_model, n_head, dropout,
                                              scale=scale, dropout_attn=dropout_attn,
                                              pos_embed=pos_embed)
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.out_fc = nn.Linear(d_model, tag_size)

    def forward(self, chars, mask):
        """
        :param chars: bert output. (batch_size, seq_len, hidden_size[embedding_dim])
        :param mask: (batch_size, seq_len)
        :return: logits: to crf. (batch_size, seq_len, tag_size)
        """

        chars = self.in_fc(chars)  # (batch_size, seq_len, d_model[])
        chars = self.transformer(chars, mask)  # (batch_size, seq_len, d_model)
        chars = self.fc_dropout(chars)  # (batch_size, seq_len, d_model)
        chars = self.out_fc(chars)  # (batch_size, seq_len, tag_size)
        return chars
