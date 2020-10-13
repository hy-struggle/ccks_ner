# /usr/bin/env python
# coding=utf-8
"""Ensemble the result"""
import json
import argparse
from collections import Counter
from itertools import groupby
import random

from rules import filter_same_ens, special_case, special_pat_1, special_pat_2
from utils import Params

# 参数解析器
parser = argparse.ArgumentParser()
# 设定参数
parser.add_argument('--num_samples', type=int, default=400, help="测试集样本数")
parser.add_argument('--vote_threshold', type=int, default=6, help="投票阈值")
parser.add_argument('--type_threshold', type=int, default=6, help="投票阈值")


def _d_overlap(overlap_ll, ne_list, re_list, handle, type_threshold=3):
    # 重叠实体
    print(f'overlap_{handle}', overlap_ll)
    # 出现最多的重叠实体
    max_c = max(overlap_ll, key=lambda t: t[1])
    overlap_ll = sorted(overlap_ll, key=lambda t: t[1], reverse=True)

    # 类别重叠时，如果频次相等，则不处理
    if handle == 'type' and overlap_ll[0][1] == overlap_ll[1][1]:
        # 要保留的实体
        retain_ens = [item[0] for item in overlap_ll]
        re_list.extend(retain_ens)
    # 类别频次不相等时保留出现次数大于阈值的
    elif handle == 'type' and overlap_ll[0][1] != overlap_ll[1][1]:
        # 要保留的实体
        retain_ens = [item[0] for item in overlap_ll if item[1] >= type_threshold]
        re_list.extend(retain_ens)
        print(f'select_{handle}:', retain_ens)
    # 边界重叠
    else:
        # 边界重叠选出现次数最多的
        print(f'select_{handle}:', max_c)
        re_list.append(max_c[0])
        # 丢掉overlap中不要的实体
        ne_list = list(set(ne_list).difference(overlap_ll))
        ne_list.append(max_c)
    return ne_list, re_list


def del_overlap(ne_list, handle=None, type_threshold=3):
    """去除融合后单样本中的重叠实体
    Args:
        ne_list (List[((ne_type, start, end), counter)]): 单样本实体集
        handle (str): 选择去重的类型，type,boundary或all
        type_threshold (int): 实体投票阈值

    Returns:
        new_list (List[((ne_type, start, end), counter)]): 去重叠后的实体集

    Examples:
        >>> c_list = [(('任务场景', 150, 155),1), (('试验要素', 150, 162),2), (('试验要素', 157, 162),1)]
        >>> del_overlap(c_list, handle='boundary')
        [('ABC', 4, 7)]
    """
    re_list = []
    while len(ne_list) != 0:
        # 按跨度降序排列（从跨度小的实体开始遍历）
        ne_list = sorted(ne_list, key=lambda ne: ne[0][2] - ne[0][1], reverse=True)
        # pop跨度小的实体
        c = ne_list.pop()
        # current start and end
        start = int(c[0][1])
        end = int(c[0][2])
        # 重叠的实体
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
            ne_list, re_list = _d_overlap(overlap_type, ne_list, re_list, handle='type',
                                          type_threshold=type_threshold)
        # 处理边界重叠实体
        elif len(overlap_boundary) > 1 and handle in ('boundary', 'all'):
            ne_list, re_list = _d_overlap(overlap_boundary, ne_list, re_list, handle='boundary',
                                          type_threshold=type_threshold)
        # 无重叠
        else:
            re_list.append(c[0])
    return list(set(re_list))


def get_union_set(num_samples, result_files, params):
    """获取所有结果
    Args:
        num_samples: 结果样本数
        result_files: list of file name
    """
    ensemble_entity = [[] for _ in range(num_samples)]
    for file_name in result_files:
        with open(params.root_path / 'ensemble/all_single_results' / file_name, 'r', encoding='utf-8') as f:
            pre = json.load(f)
            for idx in range(num_samples):
                entities = [(en['label_type'], int(en['start_pos']), int(en['end_pos']))
                            for en in pre[f'test_{idx + 1}.json']]
                ensemble_entity[idx].extend(entities)
    return ensemble_entity


def vote(ner_files, num_samples, entity_threshold, type_threshold, params, handle=None):
    """融合模型结果
    :param ner_files: list of ner result files' name
    :param num_samples: 结果样本数
    :param entity_threshold: 实体投票阈值
    """

    # get test set sentences
    with open(params.data_dir / 'test/sentences.txt', 'r', encoding='utf-8') as f_sen:
        test_sens = [line.strip().split(' ') for line in f_sen]

    # 去NER所有结果
    ensemble_entity = get_union_set(num_samples, ner_files, params)
    # counter
    counters = [Counter(s).most_common() for s in ensemble_entity]

    # 投票融合
    ner_result = [[c for c in c_list if c[1] >= entity_threshold] for c_list in counters]
    # 去重叠实体
    ner_result = [del_overlap(c_list, handle=handle, type_threshold=type_threshold) for idx, c_list in
                  enumerate(ner_result)]

    # write to json file
    submit = {}
    for idx, entities in enumerate(ner_result):
        sample4rule = []
        sample_list = []
        # 获取模型结果
        for entity in set(entities):
            content = "".join(test_sens[idx])[entity[1] - 1: entity[2]]
            en_con = (entity[0], content, entity[1], entity[2])
            # 矫正special case
            en_con = special_case(en_con, test_sens[idx])
            if not en_con:
                continue
            sample4rule.append(en_con)

        after_rule = filter_same_ens(sample4rule, ''.join(test_sens[idx]))
        after_rule = special_pat_2(special_pat_1(after_rule, test_sens[idx]), test_sens[idx])

        for entity in after_rule:
            enti_dict = {
                "label_type": entity[0],
                "overlap": 0,
                "start_pos": entity[2],
                "end_pos": entity[3]
            }
            sample_list.append(enti_dict)
        submit[f"test_{idx + 1}.json"] = sample_list

    with open(params.root_path / 'ensemble/final_sub.json', 'w', encoding='utf-8') as f_sub:
        # convert dict to json
        json_data = json.dumps(submit, indent=4, ensure_ascii=False)
        f_sub.write(json_data)


if __name__ == '__main__':
    random.seed(2020)
    args = parser.parse_args()
    params = Params()
    ner_files = ['ex1.json', 'ex2.json', 'ex3.json', 'ex4.json', 'ex5.json',
                 'ex1_256.json', 'ex2_256.json', 'ex3_256.json', 'ex4_256.json', 'ex5_256.json']
    ner_files.extend([f'ex{idx}.json' for idx in range(14, 24)])
    vote(ner_files, args.num_samples, params=params, handle='all', entity_threshold=args.vote_threshold,
         type_threshold=args.type_threshold)
