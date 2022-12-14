import unittest

import multilinear_algebra as ma


class TestOperation(unittest.TestCase):

    def get_values(self):
        self.values[(0, 0)] = 1
        self.values[(0, 1)] = 2
        self.values[(1, 0)] = 3
        self.values[(1, 1)] = 4
        return self

    def test_addition_equalType(self):
        A = ma.MLA(tensor_type=['^_'], name='A', dim=2)
        A = TestOperation.get_values(A)
        B = A + A
        B_list = [i_val for i_val in B.values.values()]
        self.assertEqual(B_list, [2, 4, 6, 8])

    def test_addition_nonEqualType(self):
        pass


if __name__ == '__main__':
    unittest.main()
