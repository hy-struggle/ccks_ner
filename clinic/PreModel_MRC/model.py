#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""Downstream task model

Examples:
    tokenizer = BertTokenizer(vocab_file=os.path.join(params.bert_model_dir, 'vocab.txt'), do_lower_case=True)
    config = BertConfig.from_json_file(os.path.join(params.bert_model_dir, 'bert_config.json'))
    model = BertModel.from_pretrained(config=config, pretrained_model_name_or_path=model_path)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from NEZHA.model_NEZHA import BertPreTrainedModel, NEZHAModel
from utils import initial_parameter


class MultiLossLayer(nn.Module):
    """implementation of "Multi-Task Learning Using Uncertainty
    to Weigh Losses for Scene Geometry and Semantics"
    """
    def __init__(self, num_loss):
        """
        Args:
            num_loss (int): number of multi-task loss
        """
        super(MultiLossLayer, self).__init__()
        # sigmas^2 (num_loss,)
        # uniform init
        self.sigmas_sq = nn.Parameter(nn.init.uniform_(torch.empty(num_loss), a=0.2, b=1.0), requires_grad=True)

    def get_loss(self, loss_set):
        """
        Args:
            loss_set (Tensor): multi-task loss (num_loss,)
        """
        # 1/2σ^2
        factor = torch.div(1.0, torch.mul(2.0, self.sigmas_sq))  # (num_loss,)
        # loss part
        loss_part = torch.sum(torch.mul(factor, loss_set))  # (num_loss,)
        # regular part
        regular_part = torch.sum(torch.log(self.sigmas_sq))
        loss = loss_part + regular_part
        return loss


class BertQueryNER(BertPreTrainedModel):
    def __init__(self, config, params):
        super(BertQueryNER, self).__init__(config)
        # pretrain model layer
        self.bert = NEZHAModel(config)

        # start and end position layer
        self.start_outputs = nn.Linear(config.hidden_size, 2)
        self.end_outputs = nn.Linear(config.hidden_size, 2)

        # 动态权重
        self.fusion_layers = params.fusion_layers
        self.dym_weight = nn.Parameter(torch.ones((self.fusion_layers, 1, 1, 1)),
                                       requires_grad=True)

        # self-adaption weight loss
        self.multi_loss_layer = MultiLossLayer(num_loss=2)

        # init weights
        self.apply(self.init_bert_weights)
        self.init_param()

    def init_param(self):
        initial_parameter(self.start_outputs)
        initial_parameter(self.end_outputs)
        nn.init.xavier_normal_(self.dym_weight)

    def get_dym_layer(self, outputs):
        """
        获取动态权重融合后的bert output(num_layer维度)
        :param outputs: origin bert output
        :return sequence_output: 融合后的bert encoder output. (batch_size, seq_len, hidden_size[embedding_dim])
        """
        hidden_stack = torch.stack(outputs[0][-self.fusion_layers:],
                                   dim=0)  # (bert_layer, batch_size, sequence_length, hidden_size)
        sequence_output = torch.sum(hidden_stack * self.dym_weight,
                                    dim=0)  # (batch_size, seq_len, hidden_size[embedding_dim])
        return sequence_output

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                start_positions=None, end_positions=None):
        """
        Args:
            start_positions: (batch x max_len x 1)
                [[0, 1, 0, 0, 1, 0, 1, 0, 0, ], [0, 1, 0, 0, 1, 0, 1, 0, 0, ]] 
            end_positions: (batch x max_len x 1)
                [[0, 1, 0, 0, 1, 0, 1, 0, 0, ], [0, 1, 0, 0, 1, 0, 1, 0, 0, ]] 
        """
        # pretrain model
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )  # sequence_output, pooled_output, (hidden_states), (attentions)

        # BERT融合
        sequence_output = self.get_dym_layer(outputs)  # (batch_size, seq_len, hidden_size[embedding_dim])

        # sequence_output = outputs[0]  # batch x seq_len x hidden
        batch_size, seq_len, hid_size = sequence_output.size()

        # get logits
        start_logits = self.start_outputs(sequence_output)  # batch x seq_len x 2
        end_logits = self.end_outputs(sequence_output)  # batch x seq_len x 2

        # train
        if start_positions is not None and end_positions is not None:
            # s & e loss
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            start_loss = loss_fct(start_logits.view(-1, 2), start_positions.view(-1))  # (bs*max_len,)
            end_loss = loss_fct(end_logits.view(-1, 2), end_positions.view(-1))
            # mask
            start_loss = torch.sum(start_loss * token_type_ids.view(-1)) / batch_size
            end_loss = torch.sum(end_loss * token_type_ids.view(-1)) / batch_size
            # total loss
            total_loss = self.multi_loss_layer.get_loss(torch.cat([start_loss.view(1), end_loss.view(1)]))
            return total_loss
        # inference
        else:
            start_pre = torch.argmax(F.softmax(start_logits, -1), dim=-1)
            end_pre = torch.argmax(F.softmax(end_logits, -1), dim=-1)
            return start_pre, end_pre


if __name__ == '__main__':
    from transformers import BertConfig
    import utils
    import os

    params = utils.Params()
    # Prepare model
    bert_config = BertConfig.from_json_file(os.path.join(params.bert_model_dir, 'bert_config.json'))
    model = BertQueryNER.from_pretrained(config=bert_config, pretrained_model_name_or_path=params.bert_model_dir,
                                         params=params)
    # 保存bert config
    model.to(params.device)

    for n, _ in model.named_parameters():
        print(n)
