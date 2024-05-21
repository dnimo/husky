from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer
import gzip
import json
import pickle

def d2p(d, f):
    with open(f, 'wb') as f:
        pickle.dump(d, f)


def p2d(f):
    with open(f, 'rb') as f:
        return pickle.load(f)


def d2j(d, f):
    with open(f, 'wt') as f:
        json.dump(d, f, indent=2)


def j2d(f):
    with open(f, 'rt') as f:
        return json.load(f)


def d2jz(d, f):
    with gzip.open(f, 'wb') as f:
        json_data = json.dumps(d, ensure_ascii=False, indent=2)
        f.write(json_data.encode('utf-8'))


def jz2d(f):
    with gzip.open(f, 'rb') as f:
        json_data = json.load(f)
        return json.loads(json_data)

# Path: tools/Model/dataUnits/dataLoader.py

class dataloader:
    def __init__(self, dataset: Dataset, batch_size: int, max_length: int, tokenizer: PreTrainedTokenizer, model_type: str):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = tokenizer

    def tokenize_fn(self, example):
        context_length = self.max_length
        outputs = self.tokenizer(
            self.tokenizer.eos_token.join(example["text"]),
            truncation=True,
            return_tensors="pt",
            padding="max_length",
            max_length=context_length + 1,
            add_special_tokens=True,
            paddding="max_length",
            return_attention_mask=True,
        )
        return {"input_ids": outputs["input_ids"][:-1], "attention_mask": outputs["attention_mask"][:-1]}

    def format_examples(self, examples):
        chunked_examples = []
        for example in examples:
            text = example["text"]
            for i in range(0, len(text), self.max_length):
                chunked_examples.append({"text": text[i : i + self.max_length]})
        return chunked_examples