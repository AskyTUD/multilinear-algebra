import unittest

import multilinear_algebra as ma
import numpy as np


class TestOperation(unittest.TestCase):

    def setUp(self) -> None:
        self.scalar1 = ma.MLA.scalar(2)
        self.scalar2 = ma.MLA.scalar(-4)
        self.objA1_ul = ma.MLA(tensor_type=['^_'], name='A1', dim=2)
        self.objA1_ul.values[(0, 0)] = 1.0
        self.objA1_ul.values[(0, 1)] = 2.5
        self.objA1_ul.values[(1, 0)] = 3.2
        self.objA1_ul.values[(1, 1)] = 4.0
        self.objA2_ul = ma.MLA(tensor_type=['^_'], name='A2', dim=2)
        self.objA2_ul.values[(0, 0)] = -1.0
        self.objA2_ul.values[(0, 1)] = 5.2
        self.objA2_ul.values[(1, 0)] = -3.6
        self.objA2_ul.values[(1, 1)] = 10.0
        self.objA3_ll = ma.MLA(tensor_type=['__'], name='A3', dim=2)
        self.objA3_ll.values[(0, 0)] = 1.0
        self.objA3_ll.values[(0, 1)] = 5.2
        self.objA3_ll.values[(1, 0)] = 3.6
        self.objA3_ll.values[(1, 1)] = 5.4
        self.objA4_ll = ma.MLA(tensor_type=['^_'], name='A4', dim=3)

    @ staticmethod
    def is_equal(a_val, b_val):
        return [True if abs(i_a - i_b) < 1e-8 else False for i_a, i_b in zip(a_val, b_val)]

    def test_dimension(self):
        pass

    def test_equality(self):
        self.assertTrue(self.objA2_ul == self.objA2_ul)
        self.assertTrue(self.objA2_ul != self.objA1_ul)
        self.assertTrue(self.objA2_ul != self.objA3_ll)

    def test_initialization(self):
        attr_set = {'dimension', 'index_order', 'indices', 'name', 'name_components', 'type', 'values'}
        for item in self.objA1_ul.__dict__:
            self.assertIn(item, attr_set)

    def test_negative_val(self):
        check0 = -self.scalar1
        check1 = -self.objA2_ul
        check1_list = [i_val for i_val in check1.values.values()]

        self.assertEqual(check0.val(), -2.0)
        self.assertTrue(all(TestOperation.is_equal(check1_list, [1.0, -5.2, 3.6, -10.0])))

    def test_addition_subtraction(self):
        S1 = self.scalar1 + self.scalar1
        S2 = self.scalar2 - self.scalar1

        B1 = self.objA1_ul + self.objA1_ul
        B1_list = [i_val for i_val in B1.values.values()]
        B2 = -self.objA1_ul + self.objA2_ul
        B2_list = [i_val for i_val in B2.values.values()]

        B3 = self.scalar1*self.objA1_ul - self.objA2_ul
        B3_list = [i_val for i_val in B3.values.values()]
        B4 = self.objA1_ul - self.objA2_ul
        B4_list = [i_val for i_val in B4.values.values()]

        self.assertEqual(S1.val(), 4.0)
        self.assertEqual(S2.val(), -6.0)

        np.testing.assert_allclose(B1_list, [2.0, 5.0, 6.4, 8.0])
        np.testing.assert_allclose(B2_list, [-2.0, 2.7, -6.8, 6.0])
        # self.assertTrue(all(TestOperation.is_equal(B1_list, [2.0, 5.0, 6.4, 8.0])))
        # self.assertTrue(all(TestOperation.is_equal(B2_list, [-2.0, 2.7, -6.8, 6.0])))

        np.testing.assert_allclose(B3_list, [3.0, -0.2, 10.0, -2.0])
        np.testing.assert_allclose(B4_list, [2.0, -2.7, 6.8, -6.0])
        # self.assertTrue(all(TestOperation.is_equal(B3_list, [3.0, -0.2, 10.0, -2.0])))
        # self.assertTrue(all(TestOperation.is_equal(B4_list, [2.0, -2.7, 6.8, -6.0])))

    def test_addition_subtraction_error(self):
        with self.assertRaises(TypeError):
            self.objA1_ul + self.objA3_ll
            self.objA3_ll + self.objA4_ll
        with self.assertRaises(TypeError):
            self.objA1_ul.id('ab') + self.objA1_ul.id('cd')
        with self.assertRaises(TypeError):
            self.objA1_ul.id('ab') - self.objA1_ul.id('cd')

    def test_naming(self):
        self.assertEqual((self.objA1_ul.id('cf')+self.objA1_ul.id('cf')).name, '(A1+A1)')
        test = self.objA1_ul.id('cf') + self.objA1_ul.id('cf')
        test.rename('newVal')
        self.assertEqual(test.name, 'newVal')

    def test_indexing(self):
        self.assertEqual(self.objA1_ul.id('ab').indices, ['a', 'b'])

    def test_scalar_value(self):
        self.assertEqual(self.scalar1.val(), 2.0)
        with self.assertRaises(TypeError):
            self.objA1_ul.val()
        x = [1e-5, 1e-3, 1e-1]
        y = np.arccos(np.cos(x))
        np.testing.assert_allclose(x, y, rtol=1e-5, atol=0)



if __name__ == '__main__':
    unittest.main()
