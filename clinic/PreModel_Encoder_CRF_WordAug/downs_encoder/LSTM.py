# /usr/bin/env python
# coding=utf-8
"""bi-lstm"""
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, tag_size, embedding_size, hidden_size, num_layers, dropout, with_ln):
        super(BiLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # set multi-lstm dropout
        self.multi_dropout = 0. if num_layers == 1 else dropout
        self.bilstm = nn.LSTM(input_size=embedding_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              dropout=self.multi_dropout,
                              bidirectional=True)

        self.with_ln = with_ln
        if with_ln:
            self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.hidden2tag = nn.Linear(hidden_size * 2, tag_size)

        self.reset_params()

    def reset_params(self):
        nn.init.xavier_normal_(self.hidden2tag.weight)

    def get_lstm_features(self, embed, mask):
        """
        :param embed: (seq_len, batch_size, embedding_size)
        :param mask: (seq_len, batch_size)
        :return lstm_features: (seq_len, batch_size, tag_size)
        """
        embed = self.dropout(embed)
        max_len, batch_size, embed_size = embed.size()
        embed = nn.utils.rnn.pack_padded_sequence(embed, mask.sum(0).long(), enforce_sorted=False)
        lstm_output, _ = self.bilstm(embed)  # (seq_len, batch_size, hidden_size*2)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, total_length=max_len)
        lstm_output = lstm_output * mask.unsqueeze(-1)
        if self.with_ln:
            lstm_output = self.layer_norm(lstm_output)
        lstm_features = self.hidden2tag(lstm_output) * mask.unsqueeze(-1)  # (seq_len, batch_size, tag_size)
        return lstm_features
