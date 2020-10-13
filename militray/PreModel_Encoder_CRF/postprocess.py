# /usr/bin/env python
# coding=utf-8

import json
import argparse
from collections import Counter

import pandas as pd

from utils import Params, IO2STR
from metrics import get_entities

# 参数解析器
parser = argparse.ArgumentParser()
parser.add_argument('--ex_index', type=int, default=3, help="实验名称index")
parser.add_argument('--num_fold', type=int, default=5, help="number of k-fold")
parser.add_argument('--fold_id', type=int, default=0, help="cross-valid index")
parser.add_argument('--mode', type=str, default='test', help="后处理结果类型")
parser.add_argument('--num_samples', type=int, default=400, help="测试集样本数")
parser.add_argument('--threshold', type=int, default=2, help="投票阈值")


def apply_fn(group):
    """恢复分组后的样本
    """
    result = []
    # 获取该组的所有实体
    for tags, s2o in zip(group.tags, group.split_to_ori):
        entities = get_entities(eval(tags))
        for entity in entities:
            result.append((entity[0], eval(s2o)[entity[1]], eval(s2o)[entity[2]]))
    return result


def postprocess(params, mode):
    # get df
    pre_df = pd.read_csv(params.params_path / f'{mode}_tags_pre.csv', encoding='utf-8')
    pre_df = pd.DataFrame(pre_df.groupby('example_id').apply(apply_fn), columns=['entities']).reset_index()

    # write to json file
    submit = {}
    for idx, entities in enumerate(pre_df['entities']):
        sample_list = []
        for entity in set(entities):
            enti_dict = {}
            enti_dict["label_type"] = IO2STR[entity[0].strip()]
            enti_dict["start_pos"] = entity[1] + 1
            enti_dict["end_pos"] = entity[2] + 1
            sample_list.append(enti_dict)
        submit[f"test_{idx + 1}.json"] = sample_list

    with open(params.params_path / f'submit_{mode}.txt', 'w', encoding='utf-8') as f_sub:
        # convert dict to json
        json_data = json.dumps(submit, indent=4, ensure_ascii=False)
        f_sub.write(json_data)


def fold_ensemble(params, num_fold, mode, num_samples, threshold):
    """五折结果融合
    """
    ensemble_entity = [[] for _ in range(num_samples)]
    for idx in range(num_fold):
        file_dir = params.params_path.parent / f'fold{idx}'
        with open(file_dir / f'submit_{mode}.txt', 'r', encoding='utf-8') as f:
            pre = json.load(f)
            for idx in range(num_samples):
                entities = [(en['label_type'], en['start_pos'], en['end_pos'])
                            for en in pre[f'test_{idx + 1}.json']]
                ensemble_entity[idx].extend(entities)
    # counter
    counters = [Counter(s).most_common() for s in ensemble_entity]
    # 融合策略(vote)
    ner_result = [[c for c in c_list if c[1] >= threshold] for c_list in counters]
    # 融合后去重叠实体
    ner_result = [[c[0] for c in c_list] for c_list in ner_result]

    # write to json file
    submit = {}
    for idx, entities in enumerate(ner_result):
        sample_list = []
        for entity in set(entities):
            enti_dict = {}
            enti_dict["label_type"] = entity[0]
            enti_dict["overlap"] = 0
            enti_dict["start_pos"] = entity[1]
            enti_dict["end_pos"] = entity[2]
            sample_list.append(enti_dict)
        submit[f"test_{idx + 1}.json"] = sample_list

    with open(params.params_path.parent / 'fus_sub_over.json', 'w', encoding='utf-8') as f_sub:
        # convert dict to json
        json_data = json.dumps(submit, indent=4, ensure_ascii=False)
        f_sub.write(json_data)


if __name__ == '__main__':
    args = parser.parse_args()
    params = Params(ex_index=args.ex_index, fold_id=args.fold_id)
    postprocess(params, mode=args.mode)
    # 最后一折后处理结束时融合结果
    if args.fold_id == args.num_fold - 1:
        print(f'do ensemble for {args.num_fold} results...')
        fold_ensemble(params, args.num_fold, args.mode, args.num_samples, args.threshold)
        print('-done')
