# -!- coding: utf-8 -!-
"""get train and test dataset"""
import json
import copy
import os
from pathlib import Path

from sklearn.model_selection import KFold
import pandas as pd

from utils import EN_DICT


def get_train_data(save_dir, train_data_path, fold_num):
    """构造sentence.txt和与其对应的tags.txt
    @:param train_data (List[Dict])
    """
    # get train data
    train_data = []
    for idx in range(1, 401):
        with open(train_data_path / f'train_V2_{idx}.json', encoding='gbk') as f:
            data = json.load(f)
            train_data.append(data)

    # 保存所有的句子和标签
    data_all_text = []
    data_all_tag = []

    for data in train_data:
        # 取文本和标注
        # 去掉原文本前后的回车换行符
        # 将原文本中间的回车换行符替换成r（符合源数据标注规则）
        # 将特殊字符替换为UNK
        data_ori = list(data['originalText'].strip().replace('\r\n', '✈').replace(' ', '✈'))
        data_text = copy.deepcopy(data_ori)
        data_entities = data['entities']

        for entity in data_entities:
            # 取当前实体类别
            en_type = entity['label_type']
            # 取当前实体标注
            en_tags = EN_DICT[en_type]  # ['B-XXX', 'I-XXX']
            start_ind = entity['start_pos'] - 1
            end_ind = entity['end_pos'] - 1
            # 替换实体
            data_text[start_ind] = en_tags[0]
            data_text[start_ind + 1:end_ind + 1] = [en_tags[1] for _ in range(end_ind - start_ind)]
        # 替换非实体
        for idx, item in enumerate(data_text):
            # 如果元素不是已标注的命名实体
            if len(item) != 5:
                data_text[idx] = EN_DICT['Others']
        assert len(data_ori) == len(data_text), f'生成的标签与原文本长度不一致！'
        data_all_text.append(data_ori)
        data_all_tag.append(data_text)

    assert len(data_all_text) == len(data_all_tag), '样本数不一致！'

    # K fold
    kf = KFold(n_splits=fold_num)
    texts_se = pd.Series(data_all_text)
    tags_se = pd.Series(data_all_tag)

    for fold_id, (train_id, val_id) in enumerate(kf.split(data_all_text)):
        save_dir_train = save_dir / f'fold{fold_id}/train'
        save_dir_val = save_dir / f'fold{fold_id}/val'

        # 创建文件夹
        if not os.path.exists(save_dir_train) or not os.path.exists(save_dir_val):
            os.makedirs(save_dir_train)
            os.makedirs(save_dir_val)

        for s_dir, ids in zip((save_dir_train, save_dir_val), (train_id, val_id)):
            # 写入文件
            with open(s_dir / f'sentences.txt', 'w', encoding='utf-8') as file_sentences, \
                    open(s_dir / f'tags.txt', 'w', encoding='utf-8') as file_tags:
                # 逐行对应写入
                for sentence, tag in zip(texts_se[ids].tolist(), tags_se[ids].tolist()):
                    file_sentences.write('{}\n'.format(' '.join(sentence)))
                    file_tags.write('{}\n'.format(' '.join(tag)))


def get_testset(data_path, save_dir, fold_num):
    """获取测试集
    """
    with open(data_path, encoding='utf-8') as f:
        data = json.load(f)
    # 将特殊字符替换为UNK
    data = [(key, sen.strip().replace('\r\n', '✈').replace(' ', '✈').replace('\x1a', '✈')) for
            key, sen in data.items()]
    # 根据序号排序
    data = sorted(data, key=lambda d: int(d[0].split('_')[-1].split('.')[0]))

    # K fold
    for fold_id in range(fold_num):
        save_dir_test = save_dir / f'fold{fold_id}/test'
        if not os.path.exists(save_dir_test):
            os.makedirs(save_dir_test)
        # 写入
        with open(save_dir_test / 'sentences.txt', 'w', encoding='utf-8') as file_sentences, \
                open(save_dir_test / 'tags.txt', 'w', encoding='utf-8') as file_tags:
            # 逐行对应写入
            for _, sentence in data:
                file_sentences.write('{}\n'.format(' '.join(sentence)))
                file_tags.write('{}\n'.format(' '.join(['O'] * len(sentence))))


if __name__ == '__main__':
    train_data_path = Path('./ccks_8_data_v2/train')
    save_dir = Path('./data')
    fold_num = 5
    # get_train_data(save_dir, train_data_path, fold_num=fold_num)
    test_data_path = Path('./ccks_8_data_v2/ccks2020_8_test_data.json')
    get_testset(test_data_path, save_dir, fold_num=fold_num)
