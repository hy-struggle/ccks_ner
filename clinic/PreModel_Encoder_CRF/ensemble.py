# /usr/bin/env python
# coding=utf-8
"""Ensemble the result"""
import json
import argparse
from collections import Counter
from itertools import groupby
import copy
from rules import entity_corr, vocab_match, corr_punc

from utils import Params

# 参数解析器
parser = argparse.ArgumentParser()
# 设定参数
parser.add_argument('--num_samples', type=int, default=300, help="测试集样本数")
parser.add_argument('--vote_threshold', type=int, default=2, help="投票阈值")
parser.add_argument('--type_threshold', type=int, default=3, help="类别重叠阈值")


def _d_overlap(overlap_ll, ne_list, re_list, handle):
    print(f'overlap_{handle}:', overlap_ll)
    # 出现最多的重叠实体
    max_c = max(overlap_ll, key=lambda t: t[1])
    overlap_ll = sorted(overlap_ll, key=lambda t: t[1], reverse=True)

    # 重叠时，如果频次相等，则全部丢弃
    if handle in ('type', 'boundary') and overlap_ll[0][1] == overlap_ll[1][1]:
        print('freq_equal:', overlap_ll)
        # 丢掉overlap中不要的实体
        ne_list = list(set(ne_list).difference(overlap_ll))
    # 类别频次不相等时保留最大的
    elif handle == 'type' and overlap_ll[0][1] != overlap_ll[1][1]:
        # 要保留的实体
        re_list.append(max_c[0])
        print(f'select_{handle}:', max_c)
    # 边界重叠
    else:
        re_list.append(max_c[0])
        print(f'select_{handle}:', max_c)
        # 丢掉overlap中不要的实体
        ne_list = list(set(ne_list).difference(overlap_ll))
        ne_list.append(max_c)
    return ne_list, re_list


def del_overlap(ne_list, handle=None, type_threshold=3):
    """去除融合后单样本中的重叠实体
    Args:
        ne_list (List[((ne_type, start, end), counter)]): 单样本实体集
        handle (str): 选择去重的类型，type,boundary或all
        type_threshold (int): 类别重叠阈值，保留出现次数大于阈值的实体

    Returns:
        new_list (List[((ne_type, start, end), counter)]): 去重叠后的实体集

    Examples:
        >>> c_list = [(('任务场景', 150, 155),1), (('试验要素', 150, 162),2), (('试验要素', 157, 162),1)]
        >>> del_overlap(c_list, handle='boundary')
        [('ABC', 4, 7)]
    """
    re_list = []
    while len(ne_list) != 0:
        # 按跨度降序排列
        ne_list = sorted(ne_list, key=lambda ne: ne[0][2] - ne[0][1], reverse=True)
        c = ne_list.pop()
        # current start and end
        start = int(c[0][1])
        end = int(c[0][2])
        overlap_ll = [oth for oth in ne_list if oth[0][1] <= end and oth[0][2] >= start]
        # 加入自己
        overlap_ll.append(c)

        # 将overlap实体分为类别重叠和边界重叠
        overlap_type = []
        overlap_boundary = []
        for _, item in groupby(overlap_ll, key=lambda ol: ol[0][1:]):
            item = list(item)
            if len(item) > 1:
                overlap_type.extend(item)
            else:
                overlap_boundary.extend(item)

        # 处理类别重叠实体
        if len(overlap_type) > 1 and handle in ('type', 'all'):
            ne_list, re_list = _d_overlap(overlap_type, ne_list, re_list, handle='type')
        elif len(overlap_boundary) > 1 and handle in ('boundary', 'all'):
            ne_list, re_list = _d_overlap(overlap_boundary, ne_list, re_list, handle='boundary')
        # 无重叠
        else:
            re_list.append(c[0])
    return list(set(re_list))


def get_union_set(num_samples, result_files):
    """获取所有结果
    Args:
        num_samples: 结果样本数
        result_files: list of file name
    """
    ensemble_entity = [[] for _ in range(num_samples)]
    for file_name in result_files:
        with open(params.root_path / f'ensemble/{file_name}/fusion_submit.txt', 'r', encoding='utf-8') as f:
            data_src_li = [dict(eval(line.strip())) for line in f]
            for idx in range(num_samples):
                sample = data_src_li[idx]
                entities = [(en['label_type'], en['start_pos'], en['end_pos']) for en in sample['entities']]
                ensemble_entity[idx].extend(entities)
    return ensemble_entity, data_src_li


def vote(ner_files, num_samples, vote_threshold, type_threshold, params, handle=None):
    """融合模型结果
    :param ner_files: list of ner result files' name
    :param num_samples: 结果样本数
    :param vote_threshold: 投票阈值
    :param type_threshold: 类别重叠阈值，保留出现次数大于阈值的实体
    """
    # 取NER所有结果
    ensemble_entity, data_src_li = get_union_set(num_samples, ner_files)
    # vocab for rule2
    with open('./ensemble/med_vocab.txt', 'r', encoding='utf-8') as f:
        vocab = [line.strip().split('\t') for line in f]

    # counter
    counters = [Counter(s).most_common() for s in ensemble_entity]
    # 投票取NER结果交集
    ner_result = [[c for c in c_list if c[1] >= vote_threshold] for c_list in counters]
    # 去重叠实体
    ner_result = [del_overlap(c_list, handle=handle, type_threshold=type_threshold) for c_list in ner_result]

    with open(params.root_path / 'ensemble/fusion_submit.txt', 'w', encoding='utf-8') as w:
        # get text
        data_text = [line["originalText"] for line in data_src_li]

        # write to json file
        for entities, text in zip(ner_result, data_text):
            sample4rule = [(entity[0].strip(), text[entity[1]:entity[2]], entity[1], entity[2]) for entity in entities]
            sample_list = []
            sample4rule = vocab_match(vocab, text, sample4rule)

            for entity in sample4rule:
                if corr_punc(entity, text) is True:
                    print('unusual_entity:', entity)
                elif isinstance(corr_punc(entity, text), tuple):
                    entity = corr_punc(entity, text)
                    print('correct_punc:', entity)

                entity = entity_corr(entity, text)
                if not entity:
                    continue

                enti_dict = {
                    "label_type": entity[0],
                    "start_pos": entity[2],
                    "end_pos": entity[3]
                }
                sample_list.append(enti_dict)

            json.dump({
                "originalText": text,
                "entities": sample_list
            }, w, ensure_ascii=False)
            w.write('\n')


if __name__ == '__main__':
    args = parser.parse_args()
    params = Params()
    ner_files = []
    ner_files.extend(['ex42', 'ex36', 'ex39'])
    vote(ner_files, args.num_samples, args.vote_threshold, params=params, handle='all',
         type_threshold=args.type_threshold)
