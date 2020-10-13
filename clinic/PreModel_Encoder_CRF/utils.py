# /usr/bin/env python
# coding=utf-8
"""utils"""
import logging
import os
import shutil
from itertools import chain
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.init as init

# 定义实体标注
EN_DICT = {
    '疾病和诊断': ['B-DIS', 'I-DIS'],
    '影像检查': ['B-SCR', 'I-SCR'],
    '实验室检验': ['B-LAB', 'I-LAB'],
    '手术': ['B-OPE', 'I-OPE'],
    '药物': ['B-MED', 'I-MED'],
    '解剖部位': ['B-POS', 'I-POS'],
    'Others': 'O'
}

IO2STR = {
    'DIS': '疾病和诊断',
    'SCR': '影像检查',
    'LAB': '实验室检验',
    'OPE': '手术',
    'MED': '药物',
    'POS': '解剖部位'
}
STR2IO = {v: k for k, v in IO2STR.items()}

PreModelDir = {
    "NEZHA": 'nezha-large',
    "RoBERTa": 'medbert-large',
    "ELECTRA": 'electra-large'
}


class Params:
    """参数定义
    """

    def __init__(self, pre_model_type='NEZHA', ex_index=1):
        # 根路径
        self.root_path = Path(os.path.abspath(os.path.dirname(__file__)))
        # 数据集路径
        self.data_dir = self.root_path / f'data'
        # 参数路径
        self.params_path = self.root_path / f'experiments/ex{ex_index}'
        # 模型保存路径
        self.model_dir = self.root_path / f'model/ex{ex_index}'

        self.pre_model_type = pre_model_type
        # downstream encoder type
        self.ds_encoder_type = 'LSTM'
        # 预训练模型路径
        self.bert_model_dir = self.root_path.parent.parent / f'{PreModelDir[self.pre_model_type]}'
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_gpu = torch.cuda.device_count()
        # 标签列表
        self.tags = list(chain(*EN_DICT.values()))
        # 用于CRF的标签
        self.tags.extend(["<START_TAG>", "<END_TAG>"])

        # 读取保存的data
        self.data_cache = True
        self.train_batch_size = 24
        self.val_batch_size = 24
        self.test_batch_size = 256

        # patience strategy
        # 最小训练次数
        self.min_epoch_num = 3
        # 容纳的提高量(f1-score)
        self.patience = 0.1
        # 容纳多少次未提高
        self.patience_num = 3

        self.seed = 2020
        # 句子最大长度(pad)
        self.max_seq_length = 128

        # BERT多层融合
        self.fusion_layers = 4
        # learning_rate
        self.fin_tuning_lr = 2e-5
        self.downs_en_lr = 1e-4
        self.crf_lr = self.fin_tuning_lr * 1000
        # 梯度截断
        self.clip_grad = 2.
        # dropout prob
        self.drop_prob = 0.3
        # 权重衰减系数
        self.weight_decay_rate = 0.01
        self.warmup_prop = 0.1
        self.gradient_accumulation_steps = 2

        # lstm hidden size
        self.lstm_hidden = 256
        # lstm layer num
        self.lstm_layer = 1

        # tener layers
        self.num_layers = 1
        # tener hidden size
        self.tener_hs = 256
        # tener head num
        self.num_heads = 4

        # rtrans
        self.k_size, self.rtrans_heads = 10, 4

    def get(self):
        """Gives dict-like access to Params instance by `params.show['learning_rate']"""
        return self.__dict__

    def load(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        """保存配置到json文件
        """
        params = {}
        with open(json_path, 'w') as f:
            for k, v in self.__dict__.items():
                if isinstance(v, (str, int, float, bool)):
                    params[k] = v
            json.dump(params, f, indent=4)


class RunningAverage:
    """A simple class that maintains the running average of a quantity
    记录平均损失

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(save=False, log_path=None):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if save and not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))

    if not logger.handlers:
        if save:
            # Logging to a file
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
            logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains the entire model, may contain other keys such as epoch, optimizer
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.makedirs(checkpoint)
    torch.save(state, filepath)
    # 如果是最好的checkpoint则以best为文件名保存
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, optimizer=True):
    """Loads entire model from file_path. If optimizer is True, loads
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        optimizer: (bool) resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ValueError("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))

    if optimizer:
        return checkpoint['model'], checkpoint['optim']
    return checkpoint['model']


def initial_parameter(net, initial_method=None):
    r"""A method used to initialize the weights of PyTorch models.

    :param net: a PyTorch model or a List of Pytorch model
    :param str initial_method: one of the following initializations.

            - xavier_uniform
            - xavier_normal (default)
            - kaiming_normal, or msra
            - kaiming_uniform
            - orthogonal
            - sparse
            - normal
            - uniform

    """
    if initial_method == 'xavier_uniform':
        init_method = init.xavier_uniform_
    elif initial_method == 'xavier_normal':
        init_method = init.xavier_normal_
    elif initial_method == 'kaiming_normal' or initial_method == 'msra':
        init_method = init.kaiming_normal_
    elif initial_method == 'kaiming_uniform':
        init_method = init.kaiming_uniform_
    elif initial_method == 'orthogonal':
        init_method = init.orthogonal_
    elif initial_method == 'sparse':
        init_method = init.sparse_
    elif initial_method == 'normal':
        init_method = init.normal_
    elif initial_method == 'uniform':
        init_method = init.uniform_
    else:
        init_method = init.xavier_normal_

    def weights_init(m):
        # classname = m.__class__.__name__
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d):  # for all the cnn
            if initial_method is not None:
                init_method(m.weight.data)
            else:
                init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for w in m.parameters():
                if len(w.data.size()) > 1:
                    init_method(w.data)  # weight
                else:
                    init.normal_(w.data)  # bias
        elif m is not None and hasattr(m, 'weight') and \
                hasattr(m.weight, "requires_grad"):
            if len(m.weight.size()) > 1:
                init_method(m.weight.data)
            else:
                init.normal_(m.weight.data)
        else:
            for w in m.parameters():
                if w.requires_grad:
                    if len(w.data.size()) > 1:
                        init_method(w.data)  # weight
                    else:
                        init.normal_(w.data)  # bias
                # print("init else")

    if isinstance(net, list):
        for n in net:
            n.apply(weights_init)
    else:
        net.apply(weights_init)


class FGM:
    """扰动训练(Fast Gradient Method)"""

    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embeddings.'):
        """在embedding层中加扰动
        :param epsilon: 系数
        :param emb_name: 模型中embedding的参数名
        """
        #
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and 'LayerNorm' not in name:
                # 保存原始参数
                self.backup[name] = param.data.clone()
                # scale
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    # 扰动因子
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embeddings.'):
        """恢复扰动前的参数
        :param emb_name: 模型中embedding的参数名
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and 'LayerNorm' not in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD:
    """扰动训练(Projected Gradient Descent)"""

    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='emb.', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]
