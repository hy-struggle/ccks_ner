# /usr/bin/env python
# coding=utf-8
"""Predict"""

import argparse
import random
import logging
import os
from tqdm import tqdm

import pandas as pd
import torch

import utils
from evaluate import extract_joint_tag
from dataloader import NERDataLoader

# 参数解析器
parser = argparse.ArgumentParser()
# 设定参数
parser.add_argument('--seed', type=int, default=2020, help="random seed for initialization")
parser.add_argument('--ex_index', type=int, default=1, help="实验名称索引")
parser.add_argument('--device_id', type=int, default=3, help="使用的GPU")
parser.add_argument('--restore_file', type=str, default='best', required=False,
                    help="Optional, name of the file containing weights to reload before training")
parser.add_argument('--mode', default='test', help="'val', 'test' or 'pseudo'")


def predict(model, data_iterator, params, mode):
    """Predict entities
    """
    # set model to evaluation mode
    model.eval()

    # id2tag dict
    idx2tag = {idx: tag for idx, tag in enumerate(params.bio_tags)}
    cls_idx2tag = {idx: tag for idx, tag in enumerate(params.type_tags)}

    pre_result = pd.DataFrame()

    for batch in tqdm(data_iterator, unit='Batch', ascii=True):
        # init
        pred_tags = []
        # to device
        batch = tuple(t.to(params.device) for t in batch)
        input_ids, input_mask, _, _, split_to_ori, example_ids = batch
        split_to_ori = split_to_ori.to('cpu').numpy().tolist()  # (bs, max_len)
        example_ids = example_ids.to('cpu').numpy().tolist()  # (bs,)
        batch_size = len(example_ids)

        # inference
        with torch.no_grad():
            batch_output, cls_pre = model(input_ids, attention_mask=input_mask)

        input_mask = input_mask.to('cpu').numpy()  # (bs, seq_len)
        cls_pre = cls_pre.detach().cpu().numpy().tolist()
        # get result
        for i in range(batch_size):
            # 恢复标签真实长度
            real_len = int(input_mask[i].sum())

            # get pre label
            pre_bio = [idx2tag.get(idx) for idx in batch_output[i]]
            pre_cls = [cls_idx2tag.get(idx) for idx in cls_pre[i][:real_len]]
            pre_re = extract_joint_tag(pre_bio, pre_cls)
            pred_tags.append(pre_re)

        # append to df
        # print(len(pred_tags))
        for example_id, pred_tag, s_to_o, mask in zip(example_ids, pred_tags, split_to_ori, input_mask):
            assert mask.sum() == len(pred_tag)
            pre_result = pre_result.append({
                'example_id': int(example_id),
                'tags': pred_tag,
                'split_to_ori': s_to_o
            }, ignore_index=True)

    pre_result.to_csv(path_or_buf=params.params_path / f'{mode}_tags_pre.csv', encoding='utf-8', index=False)


if __name__ == '__main__':
    args = parser.parse_args()
    params = utils.Params(args.ex_index)
    # 设置模型使用的gpu
    torch.cuda.set_device(args.device_id)
    # 查看现在使用的设备
    print('current device:', torch.cuda.current_device())
    # 预测验证集还是测试集
    mode = args.mode
    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed

    # Set the logger
    utils.set_logger()

    # get dataloader
    dataloader = NERDataLoader(params)

    # Define the model
    logging.info('Loading the model...')
    # Reload weights from the saved file
    model, optimizer = utils.load_checkpoint(os.path.join(params.model_dir, args.restore_file + '.pth.tar'))
    model.to(params.device)
    logging.info('- done.')

    logging.info("Loading the dataset...")
    loader = dataloader.get_dataloader(data_sign=mode)
    logging.info('-done')

    logging.info("Starting prediction...")
    # Create the input data pipeline
    predict(model, loader, params, mode)
    logging.info('-done')
