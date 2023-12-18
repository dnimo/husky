import math
import random
import MeCab
import ipadic
import numpy as np

from tqdm import tqdm
from Model.common.tokenizers import SpTokenizer
from Model.pretrain_config import *
from torch.utils.data import Dataset
from __init__ import DICT_PATH

class DataFactory(object):
    def __init__(self):
        self.tokenizer = SpTokenizer(TokenizerPath)
        self.tagger = MeCab.Tagger(ipadic.MECAB_ARGS + f" -O wakati {DICT_PATH}")
        self.vocab_size = self.tokenizer._vocab_size
        self.token_pad_id = self.tokenizer._token_pad_id
        self.token_cls_id = self.tokenizer._token_start_id
        self.token_sep_id = self.tokenizer._token_end_id
        self.token_mask_id = self.tokenizer._token_mask_id

    def __token_process(self, token_id):
        """
        以80%的几率替换为[MASK]，以10%的几率保持不变，
        以10%的几率替换为一个随机token。
        """
        rand = np.random.random()
        if rand <= 0.8:
            return self.token_mask_id
        elif rand <= 0.9:
            return token_id
        else:
            return np.random.randint(0, self.vocab_size)

    def texts_to_ids(self, text):
        text_ids = []
        # 注意roberta里并不是针对每个字进行mask，而是对字或者词进行mask
        words = self.tagger.parse(text)
        words = words.split()
        for word in words:
          # text_ids首位分别是cls和sep，这里暂时去除
          word_tokes = self.tokenizer.tokenize(text=word)[1:-1]
          words_ids = self.tokenizer.tokens_to_ids(word_tokes)
          text_ids.append(words_ids)
          
        return text_ids

    def ids_to_mask(self, text_ids):
        total_ids = []
        total_masks = []
        # 为每个字或者词生成一个概率，用于判断是否mask
        mask_rates = np.random.random(len(text_ids))
        
        for i, word_id in enumerate(text_ids):
            # 为每个字生成对应概率
            total_ids.extend(word_id)
            if mask_rates[i] < MaskRate:
                # 因为word_id可能是一个字，也可能是一个词
                for sub_id in word_id:
                    total_masks.append(self.__token_process(sub_id))
            else:
                total_masks.extend([0]*len(word_id))
        
        # 每个实例的最大长度为514，因此对一个段落进行裁剪
        # 512 = 514 - 2，给cls和sep留的位置
        tmp_ids = [self.token_cls_id]
        tmp_masks = [self.token_pad_id]
        tmp_ids.extend(total_ids[0: min((SentenceLength - 2), len(total_ids))])
        tmp_masks.extend(total_masks[0: min((SentenceLength - 2), len(total_masks))])
        # 不足514的使用padding补全
        diff = SentenceLength - len(tmp_ids)
        if diff == 1:
          tmp_ids.append(self.token_sep_id)
          tmp_masks.append(self.token_pad_id)
        else:
          # 添加结束符
          tmp_ids.append(self.token_sep_id)
          tmp_masks.append(self.token_pad_id)
          # 将剩余部分padding补全
          tmp_ids.extend([self.token_pad_id] * (diff - 1))
          tmp_masks.extend([self.token_pad_id] * (diff - 1))
        instances = (tmp_ids, tmp_masks)
        return instances


class RobertaDataSet(Dataset):
    def __init__(self, texts):
      self.texts = texts
      self.roberta_data = DataFactory()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
      output = {}
      text_ids = self.roberta_data.texts_to_ids(self.texts[item])
      sample = self.roberta_data.ids_to_mask(text_ids)
      if len(sample) < 2:
        print(f"This sample is problem %s" % sample)
      token_ids = sample[0]
      mask_ids = sample[1]
      input_token_ids = self.__gen_input_token(token_ids, mask_ids)
      segment_ids = [1 if x else 0 for x in token_ids]
      output['input_token_ids'] = input_token_ids
      output['token_ids_labels'] = token_ids
      output['segment_ids'] = segment_ids
      instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
      
      return instance

    def __gen_input_token(self, token_ids, mask_ids):
        assert len(token_ids) == len(mask_ids)
        input_token_ids = []
        for token, mask in zip(token_ids, mask_ids):
            if mask == 0:
                input_token_ids.append(token)
            else:
                input_token_ids.append(mask)
        return input_token_ids


class RobertaTestSet(Dataset):
    def __init__(self, texts):
        self.tokenizer = SpTokenizer(TokenizerPath)
        self.texts = texts
        self.test_lines = []
        self.label_lines = []
        # 读取数据
        for line in self.texts:
          if line:
            line = line.strip()
            line_list = line.split('-***-')
            self.test_lines.append(line_list[1])
            self.label_lines.append(line_list[0])

    def __len__(self):
        return len(self.label_lines)

    def __getitem__(self, item):
        output = {}
        test_text = self.test_lines[item]
        label_text = self.label_lines[item]
        test_token = self.__gen_token(test_text)
        label_token = self.__gen_token(label_text)
        segment_ids = [1 if x else 0 for x in label_token]
        output['input_token_ids'] = test_token
        output['token_ids_labels'] = label_token
        output['segment_ids'] = segment_ids
        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        return instance

    def __gen_token(self, tokens):
        tar_token_ids = [101]
        tokens = list(tokens)
        tokens = tokens[:(SentenceLength - 2)]
        for token in tokens:
            token_id = self.tokenizer.token_to_id(token)
            tar_token_ids.append(token_id)
        tar_token_ids.append(102)
        if len(tar_token_ids) < SentenceLength:
            for i in range(SentenceLength - len(tar_token_ids)):
                tar_token_ids.append(0)
        return tar_token_ids