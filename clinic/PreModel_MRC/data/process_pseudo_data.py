# /usr/bin/env python
# coding=utf-8
"""利用词表给无标签样本打标"""
import re
import json

from utils import ENTITY_TYPE, EN2QUERY


def filter_chars(text):
    """过滤无用字符
    :param text: 文本
    """
    # 找出文本中所有非中，英和数字的字符
    add_chars = set(re.findall(r'[^\u4e00-\u9fa5a-zA-Z0-9]', text))
    extra_chars = set(r"""!！￥$%*（）()-——【】:：“”";；'‘’，。？,.?、/+-""")
    add_chars = add_chars.difference(extra_chars)

    # 替换特殊字符组合
    text = re.sub('{IMG:.?.?.?}', '', text)
    text = re.sub(r'<!--IMG_\d+-->', '', text)
    text = re.sub('(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', text)  # 过滤网址
    text = re.sub('<a[^>]*>', '', text).replace("</a>", "")  # 过滤a标签
    text = re.sub('<P[^>]*>', '', text).replace("</P>", "")  # 过滤P标签
    text = re.sub('<strong[^>]*>', ',', text).replace("</strong>", "")  # 过滤strong标签
    text = re.sub('<br>', ',', text)  # 过滤br标签
    text = re.sub('www.[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', text).replace("()", "")  # 过滤www开头的网址
    text = re.sub(r'\s', '', text)  # 过滤不可见字符
    text = re.sub('Ⅴ', 'V', text)

    # 清洗
    for c in add_chars:
        text = text.replace(c, '')
    return text


def get_pseudo_data_for_predict():
    """将pseudo数据转换为待预测
    """
    with open('./task1_no_val.txt', 'r', encoding='gbk') as f_test, \
            open('./unlabel_data.txt', 'r', encoding='utf-8') as f_unla:
        texts = [filter_chars(json.loads(line.strip())['originalText']) for line in f_test]
        texts += [filter_chars(line.strip()) for line in f_unla]

    with open(f'./pseudo.data', 'w', encoding='utf-8') as f:
        result = []
        # write to json
        for idx, sample in enumerate(texts):
            # construct all type
            for type_ in ENTITY_TYPE:
                # init for single type position
                position = []
                result.append({
                    'context': filter_chars(sample),
                    'entity_type': type_,
                    'query': EN2QUERY[type_],
                    'start_position': [p[0] for p in position] if len(position) != 0 else [],
                    'end_position': [p[1] for p in position] if len(position) != 0 else []
                })

        print(f'get {len(result)} pseudo samples.')
        json.dump(result, f, indent=4, ensure_ascii=False)


def add_pseudo_to_train():
    """将pseudo数据加入训练集，只加入有标签的样本
    """
    # 读pseudo data
    with open('./fusion_pseudo.txt', 'r', encoding='utf-8') as f:
        content = [dict(eval(line.strip())) for line in f]
    # 读原始train data
    with open(f'train.data', 'r', encoding='utf-8') as f:
        ori_train = json.load(f)

    # 结构化pseudo data
    result = []
    # write to json
    for idx, sample in enumerate(content[450:]):
        # construct all type
        for type_ in ENTITY_TYPE:
            # init for single type position
            position = []
            for entity in sample['entities']:
                if entity['label_type'] == type_:
                    position.append((entity['start_pos'], entity['end_pos'] - 1))
            # 只加入有标签的pseudo data
            if len(position) != 0:
                result.append({
                    'context': sample['originalText'].strip(),
                    'entity_type': type_,
                    'query': EN2QUERY[type_],
                    'start_position': [p[0] for p in position],
                    'end_position': [p[1] for p in position]
                })
    print(f'add {len(result)} pseudo samples.')

    # 将pseudo data追加到原始train data中
    with open('./train.data', 'w', encoding='utf-8') as f:
        result = ori_train + result
        json.dump(result, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    # get_pseudo_data_for_predict()
    add_pseudo_to_train()
