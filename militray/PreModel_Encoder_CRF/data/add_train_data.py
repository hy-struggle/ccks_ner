# /usr/bin/env python
# coding=utf-8
"""向训练集追加数据"""
from pathlib import Path
import json
from itertools import groupby

from utils import EN_DICT


def add_to_train():
    # load sentences and tags
    test_data_dir = Path('../data/fold0/test/')
    with open(test_data_dir / 'sentences.txt', 'r', encoding='utf-8') as f_sen:
        sentences = [line.strip().split(' ') for line in f_sen]
    with open('../ensemble/history/ex11/fusion_submit.json', 'r', encoding='utf-8') as f:
        pre = json.load(f)

    # construct tags
    tags = []
    for idx in range(1, len(pre) + 1):
        tag = ['O'] * len(sentences[idx - 1])
        entities = pre[f'validate_V2_{idx}.json']
        sample = [(entity['label_type'], entity['start_pos'], entity['end_pos']) for entity in entities]
        for k, item in groupby(sample, key=lambda en: en[1:]):
            item = list(item)
            # 丢弃类别重叠的标签
            if len(item) == 1:
                # B-XXX
                tag[item[0][1] - 1] = EN_DICT[item[0][0]][0]
                # I-XXX
                tag[item[0][1]:item[0][2]] = [EN_DICT[item[0][0]][1] for _ in range(item[0][2] - item[0][1])]
        tags.append(tag)

    # write to files
    for fold_id in range(5):
        with open(f'../data/fold{fold_id}/train/sentences.txt', 'a', encoding='utf-8') as f_sen, \
                open(f'../data/fold{fold_id}/train/tags.txt', 'a', encoding='utf-8') as f_tag:
            for idx in range(len(sentences)):
                f_sen.write(f'{" ".join(sentences[idx])}\n')
                f_tag.write(f'{" ".join(tags[idx])}\n')


if __name__ == '__main__':
    add_to_train()
