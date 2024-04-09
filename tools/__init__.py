import os
import sys

if sys.platform.startswith('win'):
    PATH_DATA = os.getcwd() + '\\tools\\data\\'
    DICT_PATH = os.path.join(PATH_DATA, "MANBYO_202106.dic")
    STOP_WORDS = os.path.join(PATH_DATA, "ja.json")
if sys.platform.startswith('darwin') or sys.platform.startswith('linux'):
    PATH_DATA = os.getcwd() + '/tools/data/'
    DICT_PATH = os.path.join(PATH_DATA, "MANBYO_202106.dic")
    STOP_WORDS = os.path.join(PATH_DATA, "ja.json")