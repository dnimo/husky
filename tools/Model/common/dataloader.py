from datasets import load_dataset, concatenate_datasets
import glob
import gzip
import json
import pickle
import datetime


# load & save function

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


# load dataset

class huskyDatasets():
    None