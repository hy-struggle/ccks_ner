# /usr/bin/env python
# coding=utf-8
import re
import copy


def _cor_punc(punc_in, punc_another, entity, text, punctuation):
    # 有单个成对标点
    if punc_in in entity[1] and punc_another not in entity[1]:
        # 如果标点在最前，矫正
        if entity[1].index(punc_in) + entity[2] == entity[2]:
            return tuple((entity[0], entity[1][1:], entity[2] + 1, entity[3]))
        # 如果标点在最后，矫正
        elif entity[1].index(punc_in) + entity[2] == entity[3] - 1:
            return tuple((entity[0], entity[1][:-1], entity[2], entity[3] - 1))
        # 在中间
        else:
            # 左半部分在中间，右半部分在最后，补全
            if punc_in in [p[0] for p in punctuation] and text[entity[3]] == punc_another:
                return tuple((entity[0], entity[1] + punc_another, entity[2], entity[3] + 1))
            # 右半部分在中间，左半部分在最前，补全
            elif punc_in in [p[1] for p in punctuation] and text[entity[2] - 1] == punc_another:
                return tuple((entity[0], punc_another + entity[1], entity[2] - 1, entity[3]))
            else:
                return True


def correct_punc(entity, text):
    """矫正单个成对标点
    Args:
        entity (tuple): ('试验要素', '可靠性指标', 147, 151)
        text (str): 实体对应的文本

    Returns:
        有单个成对标点在两端，输出校正后的实体；
        有单个成对标点在中间，两端无匹配返回True，否则输出校正后的实体；
        无单个成对标点，返回False
    """
    # 要矫正的标点
    punctuation = [('“', '”'), ('《', '》')]

    for pair in punctuation:
        a, b = pair
        if _cor_punc(a, b, entity, text, punctuation):
            return _cor_punc(a, b, entity, text, punctuation)
        if _cor_punc(b, a, entity, text, punctuation):
            return _cor_punc(b, a, entity, text, punctuation)
    return False


def special_case(entity, text):
    """实体长度为1，非汉字的实体
    Args:
        entity (Tuple): ('试验要素', '可靠性指标', 147, 151)
    Returns:
        new_en (Tuple|bool): 校正后的实体或丢弃
    """
    if entity[1][-1] == '能' and text[entity[-1]] == '力':
        new_en = (entity[0], entity[1] + '力', entity[-2], entity[-1] + 1)
    elif entity[1][:2] == '高高' and entity[1][2] == '度':
        new_en = (entity[0], entity[1][1:], entity[-2] + 1, entity[-1])
    # 长度为1且非中文
    elif entity[2] == entity[3] and len(re.findall(r'[^\u4e00-\u9fa5]', entity[1])) == 1:
        new_en = False
    else:
        new_en = entity
    if new_en != entity:
        print('实体补齐:', new_en)
    return new_en


def filter_same_ens(entities, text):
    """如果单样本出现两个相同实体，选前面的，删除后面的
    Args:
        entities (List[Tuple]): [('试验要素', '可靠性指标', 147, 151),]
        text (str): 实体对应的文本

    Returns：
        result (List[Tuple]): 校正后的结果
    """
    result = []
    # 按start索引降序排列
    entities = sorted(entities, key=lambda en: en[-2], reverse=True)
    while len(entities) != 0:
        front_en = entities.pop()

        # 标点异常的实体删除和矫正
        if correct_punc(front_en, text):
            print('del_entity:', front_en)
            continue
        elif isinstance(correct_punc(front_en, text), tuple):
            print('del_punc:', front_en)
            front_en = correct_punc(front_en, text)

        result.append(front_en)
        # 保存tmp
        tmp_entis = copy.deepcopy(entities)
        # 遍历当前实体后面的实体
        for next_en in entities:
            # 如果后面的实体与当前实体类别和内容相同，则删除
            if next_en[:2] == front_en[:2]:
                # 删除后面的实体
                tmp_entis.remove(next_en)
                print('del_next_en:', next_en)
        entities = tmp_entis
    return result


def special_pat_2(entities, text):
    text = "".join(text)
    pat2 = re.findall(r'已集成到(.+)、(.+)、(.+)以及(.+)等', text)
    result = []
    if pat2:
        for en in entities:
            for p in pat2[0]:
                if en[1] == p:
                    result.append(('任务场景', en[1], en[2], en[3]))
                    print(('任务场景', en[1], en[2], en[3]))
                    break
            else:
                result.append(en)
    else:
        result = entities
    return result


def special_pat_1(entities, text):
    text = "".join(text)
    pat1 = re.findall(r'用于打击(.+)、(.+)及(.+)，', text)
    result = []
    # 全部小于等于10
    if pat1 and not any([False if len(en_con) < 10 else True for en_con in pat1[0]]):
        for en in entities:
            for p in pat1[0]:
                if en[1] == p:
                    result.append(('任务场景', en[1], en[2], en[3]))
                    print(('任务场景', en[1], en[2], en[3]))
                    break
            else:
                result.append(en)
    else:
        result = entities
    return list(set(result))
