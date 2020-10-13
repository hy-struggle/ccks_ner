# /usr/bin/env python
# coding=utf-8
import copy


def _corr_puc(punc_in, punc_another, entity, text, punctuation):
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


def corr_punc(entity, text):
    """矫正单个成对标点
    Args:
        entity (tuple): ('试验要素', '可靠性指标', 147, 151)
        text (str): 实体对应的文本

    Returns:
        有单个成对标点在两端，输出校正后的实体；
        有单个成对标点在中间，两端无匹配返回True，否则输出校正后的实体；
        无单个成对标点，返回False
    """
    punctuation = [('“', '”'), ('《', '》')]

    for pair in punctuation:
        a, b = pair
        if _corr_puc(a, b, entity, text, punctuation):
            return _corr_puc(a, b, entity, text, punctuation)
        if _corr_puc(b, a, entity, text, punctuation):
            return _corr_puc(b, a, entity, text, punctuation)
    return False


def findall(p, s):
    """Yields all the positions of the pattern p in the string s.
    :param p: sub str
    :param s: father str
    :return (start position, end position)
    """
    i = s.find(p)
    while i != -1:
        yield (i, i + len(p) - 1)
        i = s.find(p, i + 1)


def vocab_match(vocab, sentence, model_re):
    """词表匹配
    vocab List[List[str]]: [['药物', '奥美拉错'],]
    model_re (List[Tuple]): [('试验要素', '可靠性指标', 147, 151),]
    sentence (str)
    """
    back_up = copy.deepcopy(model_re)
    # 获取匹配的词
    match_voc = [v for v in vocab if v[1] in sentence]
    # 词典匹配结果
    vocab_re = [(v[0], v[1], pos[0], pos[1] + 1) for v in match_voc for pos in list(findall(v[1], sentence))]
    # 词典结果按实体长度排序
    vocab_re = sorted(vocab_re, key=lambda v: len(v[1]), reverse=True)

    # 词典结果与模型结果融合
    for v_re in vocab_re:
        # 如果模型结果中有重叠实体，模型较长选模型，词典较长两个都要
        overlap_m_re = [m_re for m_re in model_re if m_re[-2] < v_re[-1] and m_re[-1] > v_re[-2]]
        # 如果有重叠实体
        if overlap_m_re:
            # 模型较长
            if any([len(ov_re[1]) > len(v_re[1]) for ov_re in overlap_m_re]):
                continue
            # 词典较长
            else:
                model_re.append(v_re)
        # 没有重叠实体，加入词典结果
        else:
            model_re.append(v_re)
    fusion_re = list(set(model_re))

    pr = set(fusion_re).difference(set(back_up))
    if pr:
        print('vocab_match:', pr)
    return fusion_re


def entity_corr(entity, text):
    """实体补齐与矫正
    Args:
        entity (Tuple): ('试验要素', '可靠性指标', 147, 151)
        text (str):
    """
    if entity[1] == '白细胞' and text[entity[-1]] == '数':
        new_en = (entity[0], entity[1] + '数', entity[-2], entity[-1] + 1)
    elif entity[1] == '红细胞' and text[entity[-1]] == '数':
        new_en = (entity[0], entity[1] + '数', entity[-2], entity[-1] + 1)
    elif entity[1] == '血小板' and text[entity[-1]] == '数':
        new_en = (entity[0], entity[1] + '数', entity[-2], entity[-1] + 1)
    elif entity[1] == '右下肺肺':
        new_en = False
    elif entity[1] == '肝下':
        new_en = False
    elif entity[1][-2:] == '耳耳':
        new_en = False
    elif entity[1] == '碘油':
        new_en = False
    elif entity[1] == '头孢泊':
        new_en = (entity[0], entity[1] + '肟', entity[-2], entity[-1] + 1)
    elif entity[0] == '药物' and text[entity[-1]:entity[-1] + 3] == '注射液':
        new_en = (entity[0], entity[1] + '注射液', entity[-2], entity[-1] + 3)
    elif entity[1] == '波依定)缓释片':
        new_en = (entity[0], entity[1][:3], entity[-2], entity[-1] - 4)
    elif entity[1] == '沙坦胶囊':
        new_en = (entity[0], '缬' + entity[1], entity[-2] - 1, entity[-1])
    else:
        new_en = entity
    if new_en != entity:
        print('实体补齐:', new_en)
    return new_en
