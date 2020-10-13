# /usr/bin/env python
# coding=utf-8

from pathlib import Path
import copy
import random

from utils import EN_DICT

SEED = 2020


def get_train_val(data_path, save_dir):
    """构造sentence.txt和与其对应的tags.txt
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data_src_li = [dict(eval(line.strip())) for line in f]

    data_all_text = []
    data_all_tag = []

    for data in data_src_li:
        # 取文本和标注
        # 去掉原文本前后的回车换行符
        # 将原文本中间的回车换行符替换成UNK（符合源数据标注规则）
        # 将特殊字符替换为UNK
        data_ori = list(data['originalText'].strip().replace('\r\n', '✄').replace(' ', '✄'))
        data_text = copy.deepcopy(data_ori)
        data_entities = data['entities']

        for entity in data_entities:
            # 取当前实体类别
            en_type = entity['label_type']
            # 取当前实体标注
            en_tags = EN_DICT[en_type]  # ['B-XXX', 'I-XXX']
            start_ind = entity['start_pos']
            end_ind = entity['end_pos']
            # 替换实体
            data_text[start_ind] = en_tags[0]
            data_text[start_ind + 1:end_ind] = [en_tags[1] for _ in range(end_ind - start_ind - 1)]
        # 替换非实体
        for idx, item in enumerate(data_text):
            # 如果元素不是已标注的命名实体
            if len(item) != 5:
                data_text[idx] = EN_DICT['Others']
        # sanity check
        assert len(data_ori) == len(data_text), f'生成的标签与原文本长度不一致！'
        data_all_text.append(data_ori)
        data_all_tag.append(data_text)
    # sanity check
    assert len(data_all_text) == len(data_all_tag), '样本数不一致！'

    # shuffle
    random.seed(SEED)
    shuffle_tmp = list(zip(data_all_text, data_all_tag))
    random.shuffle(shuffle_tmp)
    data_all_text, data_all_tag = zip(*shuffle_tmp)

    # 写入训练集
    with open(save_dir / 'train/sentences.txt', 'w', encoding='utf-8') as file_sentences, \
            open(save_dir / 'train/tags.txt', 'w', encoding='utf-8') as file_tags:
        # 逐行对应写入
        for sentence, tag in zip(data_all_text[:-100], data_all_tag[:-100]):
            file_sentences.write('{}\n'.format(' '.join(sentence)))
            file_tags.write('{}\n'.format(' '.join(tag)))

    # 写入验证集
    with open(save_dir / 'val/sentences.txt', 'w', encoding='utf-8') as file_sentences, \
            open(save_dir / 'val/tags.txt', 'w', encoding='utf-8') as file_tags:
        # 逐行对应写入
        for sentence, tag in zip(data_all_text[-100:], data_all_tag[-100:]):
            file_sentences.write('{}\n'.format(' '.join(sentence)))
            file_tags.write('{}\n'.format(' '.join(tag)))


def get_testset(data_path, save_dir):
    """获取测试集
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data_src_li = [dict(eval(line.strip())) for line in f]

    # init
    all_sentences = []
    all_tags = []

    for sample in data_src_li:
        s_text = list(sample['originalText'].strip().replace('\r\n', '✄').replace(' ', '✄'))
        s_tag = [EN_DICT['Others'] for _ in range(len(s_text))]

        all_sentences.append(s_text)
        all_tags.append(s_tag)

    # 写入测试集
    with open(save_dir / 'test/sentences.txt', 'w', encoding='utf-8') as f_sen, \
            open(save_dir / 'test/tags.txt', 'w', encoding='utf-8') as f_tag:
        # 逐行写入
        for sentence, tag in zip(all_sentences, all_tags):
            f_sen.write('{}\n'.format(' '.join(sentence)))
            f_tag.write('{}\n'.format(' '.join(tag)))


if __name__ == '__main__':
    train_path = Path('./ccks2020_2_task1_train/task1_train.txt')
    test_path = Path('./ccks2_task1_val/ccks2020_2_task1_test_set_no_answer.txt')
    save_dir = Path('./data')
    get_train_val(train_path, save_dir)
    # get_testset(test_path, save_dir)
