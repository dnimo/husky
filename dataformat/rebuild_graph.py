import pandas as pd
import json
import gzip
import bz2
import polars as pl
import pickle
import glob
import joblib
import gc
import datetime

time_start = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# R/W Data

def read_json_gz(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = json.loads(f.read().decode('utf-8'))
    return data


def write_json_gz(file_path, data):
    with gzip.open(file_path, 'wb') as f:
        f.write(json.dumps(data).encode('utf-8'))


