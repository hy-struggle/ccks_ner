# /usr/bin/env python
# coding=utf-8
"""Dataloader"""

import os

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import BertTokenizer
# from NEZHA.tokenization import BertTokenizer

from dataloader_utils import read_examples, convert_examples_to_features


class WordDict(object):
    """
    Dict class to store the word
    """

    def __init__(self, wordvec_path, tokenizer, custom, max_word_in_seq=32):
        """Constructs WordDict
        Args:
            custom (bool): 是否自定义词表（随机初始化词向量）
        """
        self.wordvec_path = wordvec_path
        self.max_word_in_seq = max_word_in_seq
        self.id_to_word_list = [] if custom else ['[UNK]']
        self.word_to_id_dict = {} if custom else {'[UNK]': 0}

        with open(self.wordvec_path, "r", encoding="utf-8") as fin:
            for i, line in enumerate(fin):
                word = line.split(" ")[0]
                tokens = tuple(tokenizer.tokenize(word))
                self.id_to_word_list.append(tokens)
                self.word_to_id_dict[tokens] = i + 1
        print('WordDict has been generated!')


class FeatureDataset(Dataset):
    """Pytorch Dataset for InputFeatures
    """

    def __init__(self, features):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


class NERDataLoader(object):
    """dataloader
    """

    def __init__(self, params):
        self.params = params

        self.train_batch_size = params.train_batch_size
        self.val_batch_size = params.val_batch_size
        self.test_batch_size = params.test_batch_size

        self.data_dir = params.data_dir
        self.max_seq_length = params.max_seq_length
        self.tokenizer = BertTokenizer(vocab_file=os.path.join(params.bert_model_dir, 'vocab.txt'),
                                       do_lower_case=True)
        # 词向量
        self.word_dict = WordDict(wordvec_path=self.params.word_vec_dir, tokenizer=self.tokenizer, custom=params.custom_wordvec)
        # 保存数据(Bool)
        self.data_cache = params.data_cache

    @staticmethod
    def collate_fn(features):
        """将InputFeatures转换为Tensor
        Args:
            features (List[InputFeatures])
        Returns:
            tensors (List[Tensors])
        """
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        labels = torch.tensor([f.tag for f in features], dtype=torch.long)
        split_to_ori = torch.tensor([f.split_to_original_id for f in features], dtype=torch.long)
        example_ids = torch.tensor([f.example_id for f in features], dtype=torch.long)
        # word aug
        word_ids = torch.tensor([f.word_ids for f in features], dtype=torch.long)
        word_positions = torch.tensor([f.word_positions for f in features], dtype=torch.long)
        tensors = [input_ids, input_mask, labels, split_to_ori, example_ids, word_ids, word_positions]
        return tensors

    def get_features(self, data_sign):
        """convert InputExamples to InputFeatures
        :param data_sign: 'train', 'val' or 'test'
        :return: features (List[InputFeatures]):
        """
        print("=*=" * 10)
        print("Loading {} data...".format(data_sign))
        # get examples
        if data_sign in ("train", "val", "test", "pseudo"):
            examples = read_examples(self.data_dir, data_sign=data_sign)
        else:
            raise ValueError("please notice that the data can only be train/val/test!!")
        # get features
        # 数据保存路径
        cache_path = os.path.join(self.data_dir, "{}.cache.{}".format(data_sign, str(self.max_seq_length)))
        # 读取数据
        if os.path.exists(cache_path) and self.data_cache:
            features = torch.load(cache_path)
        else:
            # 生成数据
            features = convert_examples_to_features(self.params, examples, self.tokenizer, greed_split=False,
                                                    word_dict=self.word_dict)
            # save data
            if self.data_cache:
                torch.save(features, cache_path)
        return features

    def get_dataloader(self, data_sign="train"):
        """construct dataloader
        :param data_sign: 'train', 'val' or 'test'
        """
        # InputExamples to InputFeatures
        features = self.get_features(data_sign=data_sign)
        dataset = FeatureDataset(features)
        print(f"{len(features)} {data_sign} data loaded!")
        print("=*=" * 10)

        # construct dataloader
        # RandomSampler(dataset) or SequentialSampler(dataset)
        if data_sign == "train":
            datasampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.train_batch_size,
                                    collate_fn=self.collate_fn)
        elif data_sign == "val":
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.val_batch_size,
                                    collate_fn=self.collate_fn)
        elif data_sign in ("test", "pseudo"):
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.test_batch_size,
                                    collate_fn=self.collate_fn)
        else:
            raise ValueError("please notice that the data can only be train/val/test !!")
        return dataloader


if __name__ == '__main__':
    from utils import Params

    params = Params()
    datalodaer = NERDataLoader(params)
    f = datalodaer.get_features(data_sign='val')
    print(f[0].word_ids)
    print(f[0].word_positions)
    print(f[0].word_tuples)
