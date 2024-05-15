import unittest
from tools.Model.common.model_loaders import load_pcw_wrapper


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.model = 'cyberagent/open-calm-small'
        self.cache_dir = './cache'
        self.dataset = 'dataUnits/samples.txt'
        self.output_dir = './output'
        self.right_indentation = False
        self.n_windows = 1

    def test_something(self):
        pcw_model = load_pcw_wrapper(self.model, self.cache_dir, self.right_indentation, self.n_windows)
        # self.assertEqual(True, False)  # add assertion here
        print(pcw_model)


if __name__ == '__main__':
    unittest.main()
