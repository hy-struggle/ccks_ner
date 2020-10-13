# /usr/bin/env python
# coding=utf-8
"""Evaluate the model"""
import logging
from tqdm import tqdm
import copy
from collections import Counter

import torch

import utils
from metrics import f1_score, accuracy_score, get_entities
from metrics import classification_report


def extract_joint_tag(bio_seq, cls_seq):
    """联合两标签序列
    >>> bio_seq=['B','I','E','O']
    >>> cls_seq=['EXP','ICC','ICC','AAA']
    >>> extract_joint_tag(bio_seq, cls_seq)
    ['B-CCB', 'I-CCB', 'E-CCB', 'O']
    """
    tag_re = copy.deepcopy(bio_seq)
    entities = get_entities([f'{c}-TMP' if c != 'O' else 'O' for c in bio_seq])
    for entity in entities:
        # 实体中出现次数最多的类别
        cls = Counter(cls_seq[entity[1]:entity[2] + 1]).most_common()[0][0]
        tag_re[entity[1]] = f'B-{cls}'
        tag_re[entity[1] + 1:entity[2]] = [f'I-{cls}'] * (entity[2] - entity[1] - 1)
        tag_re[entity[2]] = f'E-{cls}'
    return tag_re


def evaluate(args, model, data_iterator, params, mark='Val', verbose=True):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()

    # id2tag dict
    idx2tag = {idx: tag for idx, tag in enumerate(params.bio_tags)}
    cls_idx2tag = {idx: tag for idx, tag in enumerate(params.type_tags)}

    true_tags = []
    pred_tags = []

    # a running average object for loss
    loss_avg = utils.RunningAverage()
    for batch in tqdm(data_iterator, unit='Batch', ascii=True):
        # to device
        batch = tuple(t.to(params.device) for t in batch)
        input_ids, input_mask, labels, type_labels, _, _ = batch

        batch_size, max_len = labels.size()

        # inference
        with torch.no_grad():
            # get loss
            loss = model(input_ids, attention_mask=input_mask.bool(), bio_labels=labels, cls_labels=type_labels)
            if params.n_gpu > 1 and args.multi_gpu:
                loss = loss.mean()  # mean() to average on multi-gpu.
            # update the average loss
            loss_avg.update(loss.item())

            # inference
            batch_output, cls_pre = model(input_ids, attention_mask=input_mask.bool())  # (bs, seq_len)

        # to list
        labels = labels.to('cpu').numpy().tolist()
        type_labels = type_labels.to('cpu').numpy().tolist()
        cls_pre = cls_pre.detach().cpu().numpy().tolist()

        # get result
        for i in range(batch_size):
            # 恢复标签真实长度
            real_len = int(input_mask[i].sum())

            # get gold label
            gold_bio = [idx2tag.get(idx) for idx in labels[i][:real_len]]
            gold_cls = [cls_idx2tag.get(idx) for idx in type_labels[i][:real_len]]
            assert len(gold_bio) == len(gold_cls), 'gold_bio not equal to gold_cls!'
            gold_re = extract_joint_tag(gold_bio, gold_cls)
            true_tags.extend(gold_re)

            # get pre label
            pre_bio = [idx2tag.get(idx) for idx in batch_output[i]]
            pre_cls = [cls_idx2tag.get(idx) for idx in cls_pre[i][:real_len]]
            assert len(pre_cls) == len(pre_bio), 'pre_cls not equal to pre_bio!'
            pre_re = extract_joint_tag(pre_bio, pre_cls)
            pred_tags.extend(pre_re)

    # sanity check
    assert len(pred_tags) == len(true_tags), 'len(pred_tags) is not equal to len(true_tags)!'

    # logging loss, f1 and report
    metrics = {}
    f1 = f1_score(true_tags, pred_tags)
    accuracy = accuracy_score(true_tags, pred_tags)
    metrics['loss'] = loss_avg()
    metrics['f1'] = f1
    metrics['accuracy'] = accuracy
    metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in metrics.items())
    logging.info("- {} metrics: ".format(mark) + metrics_str)

    # f1 classification report
    if verbose:
        report = classification_report(true_tags, pred_tags)
        logging.info(report)
    return metrics
