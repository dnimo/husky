from unitTest.medinfo_tools import scoring


import os

PATH_DATA = os.path.join(os.path.dirname(__file__), "data")

DICT_PATH = os.path.join(PATH_DATA, "MANBYO_202106.dic")
STOP_WORDS = os.path.join(PATH_DATA, "ja.json")