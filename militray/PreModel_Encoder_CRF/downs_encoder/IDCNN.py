# /usr/bin/env python
# coding=utf-8
"""idcnn"""
import torch.nn as nn
from utils import initial_parameter


class IDCNN(nn.Module):
    """
      (idcnns): ModuleList(
    (0): Sequential(
      (layer0): Conv1d(filter, filter, padding='SAME')
      (layer1): Conv1d(filter, filter, padding='SAME')
      (layer2): Conv1d(filter, filter, padding='SAME', dilation=(2,))
    )
    (1): Sequential(
      (layer0): Conv1d(filter, filter, padding='SAME')
      (layer1): Conv1d(filter, filter, padding='SAME')
      (layer2): Conv1d(filter, filter, padding='SAME', dilation=(2,))
    )
    (2): Sequential(
      (layer0): Conv1d(filter, filter, padding='SAME')
      (layer1): Conv1d(filter, filter, padding='SAME')
      (layer2): Conv1d(filter, filter, padding='SAME', dilation=(2,))
    )
   (3): Sequential(
      (layer0): Conv1d(filter, filter, padding='SAME')
      (layer1): Conv1d(filter, filter, padding='SAME')
      (layer2): Conv1d(filter, filter, padding='SAME', dilation=(2,))
    )
  )
    """

    def __init__(self, config, params, filters, tag_size, kernel_size=3, num_block=4):
        super(IDCNN, self).__init__()
        self.layer_norm = nn.LayerNorm(params.max_seq_length)
        # embedding size to filters
        self.linear = nn.Linear(config.hidden_size, filters)

        # cnn block
        self.layers = [
            {"dilation": 1},
            {"dilation": 1},
            {"dilation": 2}]
        block = nn.Sequential()
        norms_1 = nn.ModuleList([self.layer_norm for _ in range(len(self.layers))])
        for i in range(len(self.layers)):
            dilation = self.layers[i]["dilation"]
            # sanity check
            assert (dilation * (kernel_size - 1)) % 2 == 0, 'we need Lin = Lout!'
            single_block = nn.Conv1d(in_channels=filters,
                                     out_channels=filters,
                                     kernel_size=kernel_size,
                                     dilation=dilation,
                                     padding=dilation * (kernel_size - 1) // 2)
            block.add_module("layer%d" % i, single_block)
            block.add_module("relu", nn.ReLU())
            block.add_module("layernorm", norms_1[i])

        # num blocks
        self.idcnn = nn.Sequential()
        norms_2 = nn.ModuleList([self.layer_norm for _ in range(num_block)])
        for i in range(num_block):
            self.idcnn.add_module("block%i" % i, block)
            self.idcnn.add_module("relu", nn.ReLU())
            self.idcnn.add_module("layernorm", norms_2[i])

        self.hidden2tag = nn.Linear(filters, tag_size)
        # 初始化参数
        initial_parameter(self)

    def forward(self, embeddings):
        """
        :param embeddings: bert output. (batch_size, seq_len, embedding_dim)
        :return: output: idcnn output. # (batch_size, seq_len, tag_size)
        """
        embeddings = self.linear(embeddings)  # (batch_size, seq_len, filter)
        embeddings = embeddings.permute(0, 2, 1)  # (batch_size, filter, seq_len)
        output = self.idcnn(embeddings).permute(0, 2,
                                                1)  # (batch_size, filter, seq_len) -> (batch_size, seq_len, filter)
        output = self.hidden2tag(output)  # (batch_size, seq_len, tag_size)
        return output
