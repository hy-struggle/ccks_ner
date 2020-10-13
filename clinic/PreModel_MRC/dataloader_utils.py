#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dataloader.py utils"""
import re
import json


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


def whitespace_tokenize(text):
    """
    Desc:
        runs basic whitespace cleaning and splitting on a piece of text.
    """
    text = text.strip()
    # 内容为空则返回空列表
    if not text:
        return []
    tokens = list(text)
    return tokens


class InputExample(object):
    """a single set of samples of data_src
    """

    def __init__(self,
                 query_item,
                 context_item,
                 start_position=None,
                 end_position=None,
                 en_cate=None):
        self.query_item = query_item
        self.context_item = context_item
        self.start_position = start_position
        self.end_position = end_position
        self.en_cate = en_cate


class InputFeatures(object):
    """
    Desc:
        a single set of features of Examples
    Args:
        start_position: start position is a list of symbol
        end_position: end position is a list of symbol
    """

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 en_cate,
                 split_to_original_id,
                 example_id,
                 start_position=None,
                 end_position=None,
                 ):
        self.input_mask = input_mask
        self.input_ids = input_ids
        self.en_cate = en_cate
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position

        # use to split
        self.split_to_original_id = split_to_original_id
        self.example_id = example_id


def read_examples(input_file):
    """read data_src to InputExamples
    :return examples (List[InputExample])
    """
    # read json file
    with open(input_file, "r", encoding='utf-8') as f:
        input_data = json.load(f)
    # get InputExample class
    examples = []
    for entry in input_data:
        query_item = entry["query"]
        context_item = entry["context"]
        start_position = entry["start_position"]
        end_position = entry["end_position"]
        en_cate = entry["entity_type"]

        example = InputExample(query_item=query_item,
                               context_item=context_item,
                               start_position=start_position,
                               end_position=end_position,
                               en_cate=en_cate)
        examples.append(example)
    print("InputExamples:", len(examples))
    return examples


def convert_examples_to_features(params, examples, tokenizer, data_sign, pad_sign=True, greed_split=True):
    """convert InputExamples to features.
    :param examples (List[InputExamples])
    :param pad_sign: 是否补零
    :return: features (List[InputFeatures])
    """
    # tag to id
    tag2idx = {tag: idx for idx, tag in enumerate(params.tag_list)}
    features = []

    max_len = params.max_seq_length
    # 分割符
    split_pad = r'([,.!?，。！？]”?)'
    for (example_idx, example) in enumerate(examples):
        # max doc length
        max_tokens_for_doc = max_len - len(example.query_item) - 3
        # 原始文本index
        original_id = list(range(len(example.context_item)))

        if (len(example.start_position) != 0 and len(example.end_position) != 0) or data_sign != 'train':
            # split long text
            sub_contexts, starts = split_text(text=example.context_item, max_len=max_tokens_for_doc,
                                              greedy=greed_split, split_pat=split_pad)
        # 如果训练集标签为空，则不分割
        else:
            sub_contexts, starts = [example.context_item], [0]

        for context, start in zip(sub_contexts, starts):
            # 保存子文本对应原文本的位置
            split_to_original_id = original_id[start:start + len(context)]
            # tokenize query
            query_tokens = tokenizer.tokenize(example.query_item)
            # List[str]
            context_doc = whitespace_tokenize(context)
            # sanity check
            assert len(context_doc) == len(context) == len(split_to_original_id), 'check the whitespace tokenize!'

            # init
            doc_start_pos = [0] * len(context_doc)
            doc_end_pos = [0] * len(context_doc)

            # 获取文本tokens
            # 标签不为空的样本
            if len(example.start_position) != 0 and len(example.end_position) != 0:
                has_label = False
                # get gold label
                for start_item, end_item in zip(example.start_position, example.end_position):
                    # 属于该子文本的标签
                    if 0 <= (start_item - start) < len(context_doc) and 0 <= (end_item - start) < len(context_doc):
                        doc_start_pos[start_item - start] = 1
                        doc_end_pos[end_item - start] = 1
                        has_label = True
                # 如果分割后的文本没有标签，则舍弃
                if not has_label:
                    continue

            # get context_tokens
            context_doc_tokens = []
            for token in context_doc:
                # tokenize
                tmp_subword_lst = tokenizer.tokenize(token)
                if len(tmp_subword_lst) == 1:
                    context_doc_tokens.extend(tmp_subword_lst)  # context len
                else:
                    context_doc_tokens.extend(['[UNK]'])

            # cut off
            if len(context_doc_tokens) > max_tokens_for_doc:
                context_doc_tokens = context_doc_tokens[: max_tokens_for_doc]
                doc_start_pos = doc_start_pos[: max_tokens_for_doc]
                doc_end_pos = doc_end_pos[: max_tokens_for_doc]
                split_to_original_id = split_to_original_id[: max_tokens_for_doc]

            # sanity check
            assert len(context_doc_tokens) == len(doc_start_pos) == len(doc_end_pos) == len(split_to_original_id)

            # input_mask:
            #   the mask has 1 for real tokens and 0 for padding tokens.
            #   only real tokens are attended to.
            # segment_ids:
            #   segment token indices to indicate first and second portions of the inputs.
            input_tokens, segment_ids, input_mask, start_pos, end_pos, s_to_o = [], [], [], [], [], []

            input_tokens.append("[CLS]")
            for ll in (segment_ids, start_pos, end_pos):
                ll.append(0)
            input_mask.append(1)
            s_to_o.append(-1)

            # query
            input_tokens.extend(query_tokens)
            for ll in (segment_ids, start_pos, end_pos):
                ll.extend([0] * len(query_tokens))
            input_mask.extend([1] * len(query_tokens))
            s_to_o.extend([-1] * len(query_tokens))

            # sep
            input_tokens.append("[SEP]")
            for ll in (segment_ids, start_pos, end_pos):
                ll.append(0)
            input_mask.append(1)
            s_to_o.append(-1)

            # context
            input_tokens.extend(context_doc_tokens)
            segment_ids.extend([1] * len(context_doc_tokens))
            start_pos.extend(doc_start_pos)
            end_pos.extend(doc_end_pos)
            input_mask.extend([1] * len(context_doc_tokens))
            s_to_o.extend(split_to_original_id)

            # sep
            input_tokens.append("[SEP]")
            segment_ids.append(1)
            start_pos.append(0)
            end_pos.append(0)
            input_mask.append(1)
            s_to_o.append(-1)

            # token to id
            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

            # zero-padding up to the sequence length
            if len(input_ids) < params.max_seq_length and pad_sign:
                # pad
                padding = [0] * (params.max_seq_length - len(input_ids))
                # pad_token_id==0
                input_ids += padding
                input_mask += padding
                segment_ids += padding
                start_pos += padding
                end_pos += padding
                s_to_o.extend([-1] * len(padding))

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_pos,
                    end_position=end_pos,
                    en_cate=tag2idx[example.en_cate],
                    split_to_original_id=s_to_o,
                    example_id=example_idx
                ))

    return features
