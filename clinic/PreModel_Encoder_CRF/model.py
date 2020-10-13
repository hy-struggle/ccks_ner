# /usr/bin/env python
# coding=utf-8
"""model"""
import torch
import torch.nn as nn

from downs_encoder import BiLSTM, RTransformer, TENER

from NEZHA.model_NEZHA import NEZHAModel
from transformers import BertPreTrainedModel, BertModel, ElectraModel

"""表示标签开始和结束，用于CRF"""
START_TAG = "<START_TAG>"
END_TAG = "<END_TAG>"


def log_sum_exp(tensor: torch.Tensor,
                dim: int = -1,
                keepdim: bool = False) -> torch.Tensor:
    """
    Compute logsumexp in a numerically stable way.
    This is mathematically equivalent to ``tensor.exp().sum(dim, keep=keepdim).log()``.
    This function is typically used for summing log probabilities.
    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A tensor of arbitrary size.
    dim : int, optional (default = -1)
        The dimension of the tensor to apply the logsumexp to.
    keepdim: bool, optional (default = False)
        Whether to retain a dimension of size one at the dimension we reduce over.
    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


class CRFLayer(nn.Module):
    def __init__(self, tag_size, params):
        super(CRFLayer, self).__init__()

        # transition[i][j] means transition probability from j to i
        self.transition = nn.Parameter(torch.randn(tag_size, tag_size), requires_grad=True)
        self.tags = params.tags
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.tags)}
        # 重置transition参数
        self.reset_parameters()

    def reset_parameters(self):
        """重置transition参数
        """
        nn.init.xavier_normal_(self.transition)
        # initialize START_TAG, END_TAG probability in log space
        # 从i到start和从end到i的score都应该为负
        self.transition.detach()[self.tag2idx[START_TAG], :] = -10000
        self.transition.detach()[:, self.tag2idx[END_TAG]] = -10000

    def forward(self, feats, mask):
        """求total scores of all the paths
        Arg:
          feats: tag概率分布. (seq_len, batch_size, tag_size)
          mask: 填充. (seq_len, batch_size)
        Return:
          scores: (batch_size, )
        """
        seq_len, batch_size, tag_size = feats.size()
        # initialize alpha to zero in log space
        alpha = feats.new_full((batch_size, tag_size), fill_value=-10000)
        # alpha in START_TAG is 1
        alpha[:, self.tag2idx[START_TAG]] = 0

        # 取当前step的emit score
        for t, feat in enumerate(feats):
            # broadcast dimension: (batch_size, next_tag, current_tag)
            # emit_score is the same regardless of current_tag, so we broadcast along current_tag
            emit_score = feat.unsqueeze(-1)  # (batch_size, tag_size, 1)
            # transition_score is the same regardless of each sample, so we broadcast along batch_size dimension
            transition_score = self.transition.unsqueeze(0)  # (1, tag_size, tag_size)
            # alpha_score is the same regardless of next_tag, so we broadcast along next_tag dimension
            alpha_score = alpha.unsqueeze(1)  # (batch_size, 1, tag_size)
            alpha_score = alpha_score + transition_score + emit_score  # (batch_size, tag_size, tag_size)
            # log_sum_exp along current_tag dimension to get next_tag alpha
            mask_t = mask[t].unsqueeze(-1)  # (batch_size, 1)
            # 累加每次的alpha
            alpha = log_sum_exp(alpha_score, -1) * mask_t + alpha * torch.logical_not(mask_t)  # (batch_size, tag_size)
        # arrive at END_TAG
        alpha = alpha + self.transition[self.tag2idx[END_TAG]].unsqueeze(0)  # (batch_size, tag_size)

        return log_sum_exp(alpha, -1)  # (batch_size, )

    def score_sentence(self, feats, tags, mask):
        """求gold score
        Arg:
          feats: (seq_len, batch_size, tag_size)
          tags: (seq_len, batch_size)
          mask: (seq_len, batch_size)
        Return:
          scores: (batch_size, )
        """
        seq_len, batch_size, tag_size = feats.size()
        scores = feats.new_zeros(batch_size)
        tags = torch.cat([tags.new_full((1, batch_size), fill_value=self.tag2idx[START_TAG]), tags],
                         0)  # (seq_len + 1, batch_size)
        # 取一个step
        for t, feat in enumerate(feats):
            emit_score = torch.stack([f[next_tag] for f, next_tag in zip(feat, tags[t + 1])])  # (batch_size,)
            transition_score = torch.stack(
                [self.transition[tags[t + 1, b], tags[t, b]] for b in range(batch_size)])  # (batch_size,)
            # 累加
            scores += (emit_score + transition_score) * mask[t]
        # 到end的score
        transition_to_end = torch.stack(
            [self.transition[self.tag2idx[END_TAG], tag[mask[:, b].sum().long()]] for b, tag in
             enumerate(tags.transpose(0, 1))])
        scores += transition_to_end
        return scores

    def viterbi_decode(self, feats, mask):
        """维特比算法，解码最佳路径
        :param feats: (seq_len, batch_size, tag_size)
        :param mask: (seq_len, batch_size)
        :return best_path: (seq_len, batch_size)
        """
        seq_len, batch_size, tag_size = feats.size()
        # initialize scores in log space
        scores = feats.new_full((batch_size, tag_size), fill_value=-10000)
        scores[:, self.tag2idx[START_TAG]] = 0
        pointers = []

        # forward
        # 取一个step
        for t, feat in enumerate(feats):
            # broadcast dimension: (batch_size, next_tag, current_tag)
            # (bat, 1, tag_size) + (1, tag_size, tag_size)
            scores_t = scores.unsqueeze(1) + self.transition.unsqueeze(0)  # (batch_size, tag_size, tag_size)
            # max along current_tag to obtain: next_tag score, current_tag pointer
            scores_t, pointer = torch.max(scores_t, -1)  # (batch_size, tag_size), (batch_size, tag_size)
            # add emit
            scores_t += feat
            pointers.append(pointer)
            mask_t = mask[t].unsqueeze(-1)  # (batch_size, 1)
            scores = scores_t * mask_t + scores * torch.logical_not(mask_t)
        pointers = torch.stack(pointers, 0)  # (seq_len, batch_size, tag_size)
        scores += self.transition[self.tag2idx[END_TAG]].unsqueeze(0)
        best_score, best_tag = torch.max(scores, -1)  # (batch_size, ), (batch_size, )

        # backtracking
        best_path = best_tag.unsqueeze(-1).tolist()  # list shape (batch_size, 1)
        for i in range(batch_size):
            best_tag_i = best_tag[i]
            seq_len_i = int(mask[:, i].sum())
            for ptr_t in reversed(pointers[:seq_len_i, i]):
                # ptr_t shape (tag_size, )
                best_tag_i = ptr_t[best_tag_i].item()
                best_path[i].append(best_tag_i)
            # pop first tag
            best_path[i].pop()
            # reverse order
            best_path[i].reverse()
        return best_path


class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config, params):
        super().__init__(config)
        # 实体类别数
        self.num_labels = len(params.tags)

        # pretrain model
        self.pre_model_type = params.pre_model_type
        if self.pre_model_type == 'NEZHA':
            self.bert = NEZHAModel(config)
        elif self.pre_model_type == 'ELECTRA':
            self.bert = ElectraModel(config)
        elif self.pre_model_type == 'RoBERTa':
            self.bert = BertModel(config)
        else:
            raise ValueError('Pre-train Model type must be NEZHA or ELECTRA or RoBERTa!')

        # 动态权重
        self.fusion_layers = params.fusion_layers
        self.dym_weight = nn.Parameter(torch.ones((self.fusion_layers, 1, 1, 1)),
                                       requires_grad=True)

        # downstream encoder
        self.ds_encoder = params.ds_encoder_type
        if self.ds_encoder == 'LSTM':
            self.bilstm = BiLSTM(self.num_labels, embedding_size=config.hidden_size, hidden_size=params.lstm_hidden,
                                 num_layers=params.lstm_layer,
                                 dropout=params.drop_prob, with_ln=True)
        elif self.ds_encoder == 'TENER':
            self.tener = TENER(tag_size=self.num_labels, embed_size=config.hidden_size, dropout=params.drop_prob,
                               num_layers=params.num_layers, d_model=params.tener_hs, n_head=params.num_heads)
        elif self.ds_encoder == 'RTRANS':
            self.rtrans = RTransformer(tag_size=self.num_labels, dropout=params.drop_prob, d_model=config.hidden_size,
                                       ksize=params.k_size, h=params.rtrans_heads)
        else:
            raise ValueError('Downstream encoder type must be LSTM or TENER or RTRANS!')

        # crf
        self.crf = CRFLayer(self.num_labels, params)

        # init weights
        self.init_weights()
        self.reset_params()

    def reset_params(self):
        nn.init.xavier_normal_(self.dym_weight)

    def get_dym_layer(self, outputs):
        """获取动态权重融合后的BERT output.
        Args:
            outputs: origin bert output
        Returns:
            sequence_output: 融合后的bert encoder output. (bs, seq_len, hs)
        """
        if self.pre_model_type in ('ELECTRA', 'NEZHA'):
            fusion_idx = 0
        elif self.pre_model_type == 'RoBERTa':
            fusion_idx = 2
        else:
            raise ValueError('Pre-train Model type must be NEZHA or ELECTRA or RoBERTa!')

        hidden_stack = torch.stack(outputs[fusion_idx][-self.fusion_layers:],
                                   dim=0)  # (num_layers, bs, seq_len, hs)
        sequence_output = torch.sum(hidden_stack * self.dym_weight,
                                    dim=0)  # (bs, seq_len, hs)
        return sequence_output

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
    ):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: 各元素的值为0或1，避免在padding的token上计算attention。(batch_size, seq_len)
            token_type_ids: 就是token对应的句子类型id，值为0或1。为空自动生成全0。(batch_size, seq_len)
            labels: (batch_size, seq_len)
        """
        # pre-train model
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )  # sequence_output, pooled_output, (hidden_states), (attentions)

        # fusion BERT layers
        sequence_output = self.get_dym_layer(outputs)  # (batch_size, seq_len, hidden_size[embedding_dim])
        batch_size, seq_len, tag_size = sequence_output.size()

        if self.ds_encoder == 'LSTM':
            # (seq_len, batch_size, tag_size)
            feats = self.bilstm.get_lstm_features(sequence_output.transpose(1, 0), attention_mask.transpose(1, 0))
        elif self.ds_encoder == 'TENER':
            feats = self.tener(sequence_output, attention_mask)  # (bs, seq_len, tag_size)
            feats = feats.transpose(1, 0)
        elif self.ds_encoder == 'RTRANS':
            feats = self.rtrans(sequence_output, attention_mask)
            feats = feats.transpose(1, 0)
        else:
            raise TypeError('Downstream encoder type must be LSTM or TENER or RTRANS!')

        # CRF
        if labels is not None:
            # total scores
            forward_score = self.crf(feats, attention_mask.transpose(1, 0))
            gold_score = self.crf.score_sentence(feats, labels.transpose(1, 0),
                                                 attention_mask.transpose(1, 0))
            loss = (forward_score - gold_score).sum() / batch_size
            return loss
        else:
            # 维特比算法
            best_paths = self.crf.viterbi_decode(feats, attention_mask.transpose(1, 0))
            return best_paths


if __name__ == '__main__':
    from transformers import BertConfig
    import utils
    import os

    params = utils.Params()
    # Prepare model
    bert_config = BertConfig.from_json_file(os.path.join(params.bert_model_dir, 'bert_config.json'))
    model = BertForTokenClassification.from_pretrained(config=bert_config,
                                                       pretrained_model_name_or_path=params.bert_model_dir,
                                                       params=params)
    # 保存bert config
    model.to(params.device)

    for n, _ in model.named_parameters():
        print(n)
