import unittest
from tools.valuations.entropy import calculate_entropy
import torch

class MyTestCase(unittest.TestCase):
    def test_something(self):
        random = torch.rand(768)
        print(random.shape)
        en = calculate_entropy(random)
        self.assertEqual(0.56578, en)  # add assertion here


if __name__ == '__main__':
    unittest.main()
