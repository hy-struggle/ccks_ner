# /usr/bin/env python
# coding=utf-8
"""preprocess"""
from pathlib import Path
import json
import random

from utils import ENTITY_TYPE, EN2QUERY

TRAIN_DATA_DIR = Path('./ccks2020_2_task1_train')
TEST_DATA_DIR = Path('./ccks2_task1_val')
DATA_DIR = Path('./data')
SEED = 2020


def get_json_data(content, mode='train'):
    with open(DATA_DIR / f'{mode}.data', 'w', encoding='utf-8') as f:
        result = []
        # write to json
        for idx, sample in enumerate(content):
            # construct all type
            for type_ in ENTITY_TYPE:
                # init for single type position
                position = []
                if mode != 'test':
                    for entity in sample['entities']:
                        if entity['label_type'] == type_:
                            position.append((entity['start_pos'], entity['end_pos'] - 1))
                result.append({
                    'context': sample['originalText'].strip().replace('\r\n', '✄').replace(' ', '✄'),
                    'entity_type': type_,
                    'query': EN2QUERY[type_],
                    'start_position': [p[0] for p in position] if len(position) != 0 else [],
                    'end_position': [p[1] for p in position] if len(position) != 0 else []
                })

        print(f'get {len(result)} {mode} samples.')
        json.dump(result, f, indent=4, ensure_ascii=False)


def get_data_mrc():
    """获取mrc格式数据集
    """
    with open(TRAIN_DATA_DIR / 'task1_train.txt', 'r', encoding='utf-8') as f:
        data_src_li = [dict(eval(line.strip())) for line in f]

    # shuffle
    random.seed(SEED)
    random.shuffle(data_src_li)
    train_content = data_src_li[:-100]
    val_content = data_src_li[-100:]

    # get train data
    get_json_data(train_content, mode='train')
    # get val data
    get_json_data(val_content, mode='val')

    with open(TEST_DATA_DIR / 'ccks2020_2_task1_test_set_no_answer.txt', 'r', encoding='utf-8') as f:
        data_src_li = [dict(eval(line.strip())) for line in f]
    # get test data
    get_json_data(data_src_li, mode='test')


if __name__ == '__main__':
    get_data_mrc()
