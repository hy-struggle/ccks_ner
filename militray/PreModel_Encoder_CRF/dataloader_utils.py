#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dataloader.py utils"""

import re


def split_text(text, max_len, split_pat=r'([，。]”?)', greedy=False):
    """文本分片
    将超过长度的文本分片成多段满足最大长度要求的最长连续子文本
    约束条件：1）每个子文本最大长度不超过max_len；
             2）所有的子文本的合集要能覆盖原始文本。
    Arguments:
        text {str} -- 原始文本
        max_len {int} -- 最大长度

    Keyword Arguments:
        split_pat {str or re pattern} -- 分割符模式 (default: {SPLIT_PAT})
        greedy {bool} -- 是否选择贪婪模式 (default: {False})
                         贪婪模式：在满足约束条件下，选择子文本最多的分割方式
                         非贪婪模式：在满足约束条件下，选择冗余度最小且交叉最为均匀的分割方式

    Returns:
        tuple -- 返回子文本列表以及每个子文本在原始文本中对应的起始位置列表

    Examples:
        text = '今夕何夕兮，搴舟中流。今日何日兮，得与王子同舟。蒙羞被好兮，不訾诟耻。心几烦而不绝兮，得知王子。山有木兮木有枝，心悦君兮君不知。'
        sub_texts, starts = split_text(text, maxlen=30, greedy=False)
        for sub_text in sub_texts:
            print(sub_text)
        print(starts)
        for start, sub_text in zip(starts, sub_texts):
            if text[start: start + len(sub_text)] != sub_text:
            print('Start indice is wrong!')
            break
    """
    # 文本小于max_len则不分割
    if len(text) <= max_len:
        return [text], [0]
    # 分割字符串
    segs = re.split(split_pat, text)
    # init
    sentences = []
    # 将分割后的段落和分隔符组合
    for i in range(0, len(segs) - 1, 2):
        sentences.append(segs[i] + segs[i + 1])
    if segs[-1]:
        sentences.append(segs[-1])
    n_sentences = len(sentences)
    sent_lens = [len(s) for s in sentences]

    # 所有满足约束条件的最长子片段
    alls = []
    for i in range(n_sentences):
        length = 0
        sub = []
        for j in range(i, n_sentences):
            if length + sent_lens[j] <= max_len or not sub:
                sub.append(j)
                length += sent_lens[j]
            else:
                break
        alls.append(sub)
        # 将最后一个段落加入
        if j == n_sentences - 1:
            if sub[-1] != j:
                alls.append(sub[1:] + [j])
            break

    if len(alls) == 1:
        return [text], [0]

    if greedy:
        # 贪婪模式返回所有子文本
        sub_texts = [''.join([sentences[i] for i in sub]) for sub in alls]
        starts = [0] + [sum(sent_lens[:i]) for i in range(1, len(alls))]
        return sub_texts, starts
    else:
        # 用动态规划求解满足要求的最优子片段集
        DG = {}  # 有向图
        N = len(alls)
        for k in range(N):
            tmplist = list(range(k + 1, min(alls[k][-1] + 1, N)))
            if not tmplist:
                tmplist.append(k + 1)
            DG[k] = tmplist

        routes = {}
        routes[N] = (0, -1)
        for i in range(N - 1, -1, -1):
            templist = []
            for j in DG[i]:
                cross = set(alls[i]) & (set(alls[j]) if j < len(alls) else set())
                w_ij = sum([sent_lens[k] for k in cross]) ** 2  # 第i个节点与第j个节点交叉度
                w_j = routes[j][0]  # 第j个子问题的值
                w_i_ = w_ij + w_j
                templist.append((w_i_, j))
            routes[i] = min(templist)

        sub_texts, starts = [''.join([sentences[i] for i in alls[0]])], [0]
        k = 0
        while True:
            k = routes[k][1]
            sub_texts.append(''.join([sentences[i] for i in alls[k]]))
            starts.append(sum(sent_lens[: alls[k][0]]))
            if k == N - 1:
                break

    return sub_texts, starts


class InputExample(object):
    """a single set of samples of data_src
    """

    def __init__(self, sentence, tag):
        self.sentence = sentence
        self.tag = tag


class InputFeatures(object):
    """
    Desc:
        a single set of features of data_src
    """

    def __init__(self,
                 input_ids,
                 input_mask,
                 tag,
                 split_to_original_id,
                 example_id
                 ):
        self.input_mask = input_mask
        self.input_ids = input_ids
        self.tag = tag

        # use to split
        self.split_to_original_id = split_to_original_id
        self.example_id = example_id


def read_examples(data_dir, data_sign):
    """read data_src to InputExamples
    :return examples (List[InputExample])
    """
    examples = []
    # read src data
    with open(data_dir / f'{data_sign}/sentences.txt', "r", encoding='utf-8') as f_sen, \
            open(data_dir / f'{data_sign}/tags.txt', 'r', encoding='utf-8') as f_tag:
        for sen, tag in zip(f_sen, f_tag):
            example = InputExample(sentence=sen.strip().split(' '), tag=tag.strip().split(' '))
            examples.append(example)
    print("InputExamples:", len(examples))
    return examples


def convert_examples_to_features(params, examples, tokenizer, pad_sign=True, greed_split=True):
    """convert examples to features.
    :param examples (List[InputExamples])
    :param pad_sign: 是否补零
    """
    # tag to id
    tag2idx = {tag: idx for idx, tag in enumerate(params.tags)}
    features = []

    # context max len
    max_len = params.max_seq_length
    split_pad = r'([,.!?，。！？]”?)'

    for (example_idx, example) in enumerate(examples):
        # split long text
        sub_texts, starts = split_text(text=''.join(example.sentence), max_len=max_len,
                                       greedy=greed_split, split_pat=split_pad)
        original_id = list(range(len(example.sentence)))

        # get split features
        for text, start in zip(sub_texts, starts):
            # tokenize返回为空则设为[UNK]
            text_tokens = [tokenizer.tokenize(token)[0] if len(tokenizer.tokenize(token)) == 1 else '[UNK]'
                           for token in text]
            # label id
            tag_ids = [tag2idx[tag] for tag in example.tag[start:start + len(text)]]
            # 保存子文本对应原文本的位置
            split_to_original_id = original_id[start:start + len(text)]

            assert len(tag_ids) == len(split_to_original_id), 'check the length of tag_ids and split_to_original_id!'

            # cut off
            if len(text_tokens) > max_len:
                text_tokens = text_tokens[:max_len]
                tag_ids = tag_ids[:max_len]
                split_to_original_id = split_to_original_id[:max_len]
            # token to id
            text_ids = tokenizer.convert_tokens_to_ids(text_tokens)

            # sanity check
            assert len(text_ids) == len(tag_ids), f'check the length of text_ids and tag_ids!'
            assert len(text_ids) == len(split_to_original_id), f'check the length of text_ids and split_to_original_id!'

            # zero-padding up to the sequence length
            if len(text_ids) < max_len and pad_sign:
                # 补零
                pad_len = max_len - len(text_ids)
                # token_pad_id=0
                text_ids += [0] * pad_len
                tag_ids += [tag2idx['O']] * pad_len
                split_to_original_id += [-1] * pad_len

            # mask
            input_mask = [1 if idx > 0 else 0 for idx in text_ids]

            # get features
            features.append(
                InputFeatures(
                    input_ids=text_ids,
                    tag=tag_ids,
                    input_mask=input_mask,
                    split_to_original_id=split_to_original_id,
                    example_id=example_idx
                ))

    return features
