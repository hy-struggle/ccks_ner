#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""do inference"""
import argparse
import logging
import os
import random
from tqdm import tqdm

import torch
import pandas as pd

import utils
from utils import EN2QUERY
from evaluate import pointer2bio
from dataloader import NERDataLoader

# 参数解析器
parser = argparse.ArgumentParser()
# 设定参数
parser.add_argument('--seed', type=int, default=2020, help="random seed for initialization")
parser.add_argument('--ex_index', type=int, default=1, help="实验名称索引")
parser.add_argument('--device_id', type=int, default=3, help="使用的GPU")
parser.add_argument('--restore_file', default='best', required=False,
                    help="Optional, name of the file containing weights to reload before training")
parser.add_argument('--mode', default='test', help="'val', 'test' or 'pseudo'")


def predict(model, test_dataloader, params, mode):
    """
    预测并将结果输出至文件
    Args:
        mode (str): 'val' or 'test'
    """
    model.eval()
    # init
    pre_result = pd.DataFrame()

    # idx to label
    cate_idx2label = {idx: value for idx, value in enumerate(params.tag_list)}

    # get data
    for batch in tqdm(test_dataloader, unit='Batch', ascii=True):
        # to device
        batch = tuple(t.to(params.device) for t in batch)
        input_ids, input_mask, segment_ids, _, _, ne_cate, split_to_ori, example_ids = batch

        # inference
        with torch.no_grad():
            start_pre, end_pre = model(input_ids, segment_ids, input_mask)

        # predict label
        start_label = start_pre.detach().cpu().numpy().tolist()
        end_label = end_pre.detach().cpu().numpy().tolist()
        # mask
        input_mask = input_mask.to("cpu").detach().numpy().tolist()
        ne_cate = ne_cate.to("cpu").numpy().tolist()
        split_to_ori = split_to_ori.to('cpu').numpy().tolist()  # (bs, max_len)
        example_ids = example_ids.to('cpu').numpy().tolist()  # (bs,)

        # get result
        for start_p, end_p, input_mask_s, ne_cate_s, s_t_o, example_id in zip(start_label, end_label,
                                                                              input_mask, ne_cate, split_to_ori,
                                                                              example_ids):
            ne_cate_str = cate_idx2label[ne_cate_s]
            # 问题长度
            q_len = len(EN2QUERY[ne_cate_str])
            # 有效长度
            act_len = sum(input_mask_s[q_len + 2:-1])
            # 转换为BIO标注
            pre_bio_labels = pointer2bio(start_p[q_len + 2:q_len + 2 + act_len],
                                         end_p[q_len + 2:q_len + 2 + act_len],
                                         en_cate=ne_cate_str)
            # append to df
            pre_result = pre_result.append({
                'example_id': int(example_id),
                'tags': pre_bio_labels,
                'split_to_ori': s_t_o[q_len + 2:q_len + 2 + act_len]
            }, ignore_index=True)

    pre_result.to_csv(path_or_buf=params.params_path / f'{mode}_tags_pre.csv', encoding='utf-8', index=False)


if __name__ == '__main__':
    args = parser.parse_args()
    params = utils.Params(ex_index=args.ex_index)
    # Set the logger
    utils.set_logger(save=False)
    # 预测验证集还是测试集
    mode = args.mode
    # 设置模型使用的gpu
    torch.cuda.set_device(args.device_id)
    # 查看现在使用的设备
    logging.info('current device:{}'.format(torch.cuda.current_device()))

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed

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
    logging.info("- done.")

    logging.info("Starting prediction...")
    predict(model, loader, params, mode)
    logging.info('- done.')
