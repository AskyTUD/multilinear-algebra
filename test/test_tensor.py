import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import unittest

# import casadi as ca
import numpy as np

import multilinear_algebra as ma


class TestOperation(unittest.TestCase):
    """test tensor class

    Args:
        unittest (_type_): _description_
    """

    def setUp(self) -> None:
        # define scalar: version 1
        self.scalar1 = ma.Tensor(value=2, name="s")
        # define scalar: version 2
        self.scalar2 = ma.Tensor()
        self.scalar2.initialize_tensor({"name": "t", "value": 3.2})
        # define scalar: version 3
        self.scalar3 = ma.Tensor()
        self.scalar3.initialize_tensor({"name": "v"})
        self.scalar3.assign_values(value={(): 2})
        # define tensor: version 1
        self.tensor1 = ma.Tensor(
            type="^_",
            dimension=2,
            name="X",
            value={(0, 0): 2, (0, 1): -1.5, (1, 0): 2.7, (1, 1): 3.5},
        )
        # define tensor: version 2
        self.tensor2 = ma.Tensor()
        self.tensor2.initialize_tensor(
            {
                "type": "^_",
                "dimension": [2, 2],
                "name": "A",
                "value": {(0, 0): 1.5, (0, 1): -1.2, (1, 0): 2.0, (1, 1): 3.5},
            }
        )
        # define tensor: version 3
        self.tensor3 = ma.Tensor()
        self.tensor3.initialize_tensor(
            {
                "type": "^_",
                "dimension": [2, 2],
                "name": "B",
            }
        )
        self.tensor3.assign_values(value={(0, 0): 1.5, (0, 1): -1.2, (1, 0): 2.0, (1, 1): 3.5})
        #
        self.tensor_A = ma.Tensor(
            type="^_",
            dimension=2,
            name="A",
            value={(0, 0): 2, (0, 1): -1.5, (1, 0): 2.7, (1, 1): 3.5},
        )
        self.tensor_G = ma.Tensor(
            type="__",
            dimension=2,
            name="G",
            value={(0, 0): 2, (0, 1): -1.5, (1, 0): 2.7, (1, 1): 3.5},
        )

    def test_init(self):
        expected_attributes = {
            "dimension": [],
            "index_order": [],
            "indices": [],
            "name": "",
            "name_components": "",
            "type": (),
            "value": {},
            "is_initialized": False,
            "is_scalar": False,
        }
        current_attributes = vars(ma.Tensor())
        self.assertDictEqual(current_attributes, expected_attributes)

    def test_initialize_tensor(self):
        scalar1 = ma.Tensor()
        tensor1 = ma.Tensor()

        with self.assertRaises(NameError) as context1:
            scalar1.initialize_tensor({"value": 3.2})
        self.assertEqual("need to define a name!", str(context1.exception))

        with self.assertRaises(NameError) as context2:
            tensor1.initialize_tensor({"typ": "_", "name": "a"})
        self.assertEqual(
            "to initialize a tensor, use properties: type, dimension!",
            str(context2.exception),
        )

    def test_assign_values(self):

        with self.assertRaises(IndexError) as context:
            help_tensor = ma.Tensor()
            help_tensor.assign_values(value={(0, 0): 3, (0, 1): -3.5, (1, 0): 2.14, (1, 1): -0.3})
        self.assertEqual(
            "tensor is not initialized -> no tensor indices are known!",
            str(context.exception),
        )

        with self.assertRaises(IndexError) as context:
            help_tensor = ma.Tensor(type="^_", dimension=2, name="A")
            help_tensor.assign_values(value={(0, 0): 3, (0, 1): -3.5, (1, 0): 2.14, (1, 2): -0.3})
        self.assertEqual(
            "index " + str((1, 1)) + " is not an element of values!",
            str(context.exception),
        )

    def test_re_name_index(self):

        new_name = "newA"
        new_indices = "ab"
        self.tensor1.idx(new_indices)
        self.assertEqual("".join(self.tensor1.indices), new_indices)
        self.assertEqual(self.tensor1.name_components, "X^a_b")

        self.tensor1.rename(new_name)
        self.assertEqual(self.tensor1.name, new_name)
        self.assertEqual(self.tensor1.name_components, "newA^a_b")

    def test_equal(self):
        self.assertTrue(self.tensor1 == self.tensor1)
        self.assertTrue(self.tensor1 != self.tensor2)

    def test_neagtive(self):
        s1n = -self.scalar1
        np.testing.assert_equal(s1n.value[()].full(), -self.scalar1.value[()].full())
        t3n = -self.tensor3
        val1 = [i_val.full() for i_val in self.tensor3.value.values()]
        val2 = [-i_val.full() for i_val in t3n.value.values()]
        np.testing.assert_equal(val1, val2)

    def test_addition(self):
        s2 = self.scalar1 + self.scalar1
        np.testing.assert_equal(2 * self.scalar1.value[()].full(), s2.value[()].full())

        B = self.tensor_A + self.tensor_A
        val_B = [i_val.full() for i_val in B.value.values()]
        val_A = [2 * i_val.full() for i_val in self.tensor_A.value.values()]
        np.testing.assert_equal(val_A, val_B)

        with self.assertRaises(TypeError) as context:
            self.scalar1 + self.tensor_A
        self.assertEqual(
            "invalid addition of scalar and tensor!",
            str(context.exception),
        )

        with self.assertRaises(TypeError) as context:
            self.tensor_G + self.tensor_A
        self.assertEqual(
            "both tensors are not in the same space!",
            str(context.exception),
        )

        with self.assertRaises(TypeError) as context:
            self.tensor_A.idx("ab", n_t=True) + self.tensor_A.idx("cd", n_t=True)
        self.assertEqual(
            "indices of the two tensors do not match!",
            str(context.exception),
        )

    def test_subtraction(self):
        s2 = self.scalar1 - self.scalar1
        np.testing.assert_equal(0, s2.value[()].full())

        B = self.tensor_A - self.tensor_A
        val_B = [i_val.full() for i_val in B.value.values()]
        val_A = [0 * i_val.full() for i_val in self.tensor_A.value.values()]
        np.testing.assert_equal(val_A, val_B)

        with self.assertRaises(TypeError) as context:
            self.scalar1 - self.tensor_A
        self.assertEqual(
            "invalid addition of scalar and tensor!",
            str(context.exception),
        )

        with self.assertRaises(TypeError) as context:
            self.tensor_G - self.tensor_A
        self.assertEqual(
            "both tensors are not in the same space!",
            str(context.exception),
        )

        with self.assertRaises(TypeError) as context:
            self.tensor_A.idx("ab", n_t=True) - self.tensor_A.idx("cd", n_t=True)
        self.assertEqual(
            "indices of the two tensors do not match!",
            str(context.exception),
        )

    def test_multiplication(self):
        prod_A = self.tensor1.idx("ab", n_t=True) * self.tensor_A.idx("bc", n_t=True)
        val_prod_A = [np.array(i_val) for i_val in prod_A.value.values()]
        val_prod_A_ref = [np.array([[i_val]]) for i_val in [-0.05, -8.25, 14.85, 8.2]]
        np.testing.assert_allclose(val_prod_A_ref, val_prod_A)

        with self.assertRaises(TypeError) as context:
            self.tensor1.idx("ba", n_t=True) * self.tensor_A.idx("bc", n_t=True)
        self.assertEqual(
            "indices are not suitable for multiplication!",
            str(context.exception),
        )

    def test_is_same_space(self):
        pass

    def test_is_valid_indices(self):
        pass

    def test_get_indice_list(self):
        pass


if __name__ == "__main__":
    unittest.main()
