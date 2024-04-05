import unittest
from tools.valuations import Rouge


class MyTestCase(unittest.TestCase):
    def test_something(self):
        rouge = Rouge().valuate("私は猫です", "私は犬です")
        self.assertEqual(0.98, rouge)  # add assertion here


if __name__ == '__main__':
    unittest.main()
