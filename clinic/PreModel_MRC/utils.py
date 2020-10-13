#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""utils"""
import json
import os
from pathlib import Path
import shutil
import logging

import torch
import torch.nn as nn
import torch.nn.init as init

ENTITY_TYPE = ['疾病和诊断', '影像检查', '实验室检验', '手术', '药物', '解剖部位']
IO2STR = {
    'DIS': '疾病和诊断',
    'SCR': '影像检查',
    'LAB': '实验室检验',
    'OPE': '手术',
    'MED': '药物',
    'POS': '解剖部位'
}
EN2QUERY = {
    '疾病和诊断': '找出疾病和诊断,例如癌症,病变,炎症,增生,肿瘤',
    '影像检查': '找出影像检查,例如CT,CR,MRI,彩超,X光,造影',
    '实验室检验': '找出实验室检验,例如血型,比率,酶,蛋白,PH',
    '手术': '找出手术,例如切除,检查,根治,电切,穿刺,移植',
    '药物': '找出药物,例如胶囊',
    '解剖部位': '找出解剖部位,例如胃,肠,肝,肺,胸,臀'
}


class Params:
    """参数定义
    """

    def __init__(self, ex_index=1):
        """
        Args:
            ex_index (int): 实验名称索引
        """
        # 根路径
        self.root_path = Path(os.path.abspath(os.path.dirname(__file__)))
        self.data_dir = self.root_path / 'data'
        self.params_path = self.root_path / f'experiments/ex{ex_index}'
        self.bert_model_dir = self.root_path.parent.parent / 'nezha-large'
        self.model_dir = self.root_path / f'model/ex{ex_index}'

        # 读取保存的data
        self.data_cache = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_gpu = torch.cuda.device_count()

        self.train_batch_size = 64
        self.val_batch_size = 64
        self.test_batch_size = 256

        # patience策略
        self.patience = 0.1
        self.patience_num = 3
        self.min_epoch_num = 3

        # 标签列表
        self.tag_list = ENTITY_TYPE
        self.max_seq_length = 128

        self.fusion_layers = 4
        self.dropout = 0.3
        self.weight_decay_rate = 0.1
        self.fin_tuning_lr = 2e-5
        self.downstream_lr = 1e-4
        # 梯度截断
        self.clip_grad = 2
        self.warmup_prop = 0.1
        self.gradient_accumulation_steps = 2

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


def set_logger(save, log_path=None):
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
        # 递归创建目录
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
