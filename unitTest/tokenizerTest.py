import unittest
from tools.huskyTokenizers import MeCabTokenizer

class MyTestCase(unittest.TestCase):
    def test_something(self):
        mecab = MeCabTokenizer()
        text = "私は猫です"
        tokens = mecab.tokenize(text)
        print(tokens)
        self.assertEqual(['私', 'は', '猫', 'です'], tokens)  # add assertion here


if __name__ == '__main__':
    unittest.main()
