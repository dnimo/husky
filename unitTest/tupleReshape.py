import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        # 示例元组
        example_tuple = (32, 2, 1, 32, 1300)

        # 将第一维度和第二维度进行交换
        reshaped_tuple = example_tuple[1], example_tuple[0], *example_tuple[2:]

        # 打印交换后的元组
        print("Reshaped tuple:", reshaped_tuple)
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
