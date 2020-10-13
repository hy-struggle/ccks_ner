# /usr/bin/env python
# coding=utf-8
"""Evaluate the model"""
import logging
from tqdm import tqdm

import torch

import utils
from metrics import f1_score, accuracy_score
from metrics import classification_report


def evaluate(args, model, data_iterator, params, mark='Val', verbose=True):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()

    # id2tag dict
    idx2tag = {idx: tag for idx, tag in enumerate(params.tags)}

    true_tags = []
    pred_tags = []

    # a running average object for loss
    loss_avg = utils.RunningAverage()
    for batch in tqdm(data_iterator, unit='Batch', ascii=True):
        # to device
        batch = tuple(t.to(params.device) for t in batch)
        input_ids, input_mask, labels, _, _ = batch

        batch_size, max_len = labels.size()

        # inference
        with torch.no_grad():
            # get loss
            loss = model(input_ids, attention_mask=input_mask.bool(), labels=labels)
            if params.n_gpu > 1 and args.multi_gpu:
                loss = loss.mean()  # mean() to average on multi-gpu.
            # update the average loss
            loss_avg.update(loss.item())

            # inference
            batch_output = model(input_ids, attention_mask=input_mask.bool())

        # 恢复标签真实长度
        real_batch_tags = []
        for i in range(batch_size):
            real_len = int(input_mask[i].sum())
            real_batch_tags.append(labels[i][:real_len].to('cpu').numpy())

        # List[int]
        pred_tags.extend([idx2tag.get(idx) for indices in batch_output for idx in indices])
        true_tags.extend([idx2tag.get(idx) for indices in real_batch_tags for idx in indices])
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
