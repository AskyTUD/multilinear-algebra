import unittest

import multilinear_algebra as ma

#import casadi as ca
#import numpy as np



class TestOperation(unittest.TestCase):
    def setUp(self) -> None:
        self.tensor1 = ma.Tensor(type="^_", dimension=2, name="A")
        self.tensor2 = ma.Tensor()
        self.tensor2.initialize_tensor({"type": "__", "dimension": [2, 2], "name": "B"})
        self.tensor3 = ma.Tensor()
        self.tensor3.initialize_tensor(
            {
                "type": "__",
                "dimension": [2, 2],
                "name": "C",
                "values": {(0, 0): 1.0, (0, 1): 1.2, (1, 0): 2.5, (1, 1): -0.5},
            }
        )
        self.tensor4 = ma.Tensor()
        self.tensor4.initialize_tensor(
            {
                "type": "__",
                "dimension": [2, 2],
                "name": "D",
            }
        )
        self.tensor4.assign_values(
            values={(0, 0): 3, (0, 1): -3.5, (1, 0): 2.14, (1, 1): -0.3}
        )

    def test_init(self):
        expected_attributes = {
            "dimension": [],
            "index_order": [],
            "indices": [],
            "name": "",
            "name_components": "",
            "type": (),
            "values": {},
            "is_initialized": False,
        }
        current_attributes = vars(ma.Tensor())
        self.assertDictEqual(current_attributes, expected_attributes)

    def test_initialize_tensor(self):
        with self.assertRaises(Exception) as context:
            ma.Tensor(typ="_")
        self.assertEqual(
            "need to define: type, name, dimension!", str(context.exception)
        )

    def test_assign_values(self):
        with self.assertRaises(IndexError) as context:
            help_tensor = ma.Tensor()
            help_tensor.assign_values(
                values={(0, 0): 3, (0, 1): -3.5, (1, 0): 2.14, (1, 1): -0.3}
            )
        self.assertEqual(
            "tensor is not initialized -> no tensor indices are known!",
            str(context.exception),
        )

        with self.assertRaises(IndexError) as context:
            help_tensor = ma.Tensor(type="^_", dimension=2, name="A")
            help_tensor.assign_values(
                values={(0, 0): 3, (0, 1): -3.5, (1, 0): 2.14, (1, 2): -0.3}
            )
        self.assertEqual(
            "Index " + str((1, 1)) + " is not an element of values!",
            str(context.exception),
        )

    def test_re_name_index(self):
        new_name = "newA"
        new_indices = 'ab'
        self.tensor1.idx(new_indices)
        self.assertEqual(''.join(self.tensor1.indices), new_indices)
        self.assertEqual(self.tensor1.name_components, 'A^a_b')

        self.tensor1.rename(new_name)
        self.assertEqual(self.tensor1.name, new_name)
        self.assertEqual(self.tensor1.name_components, 'newA^a_b')

    def test_equal(self):
        self.assertTrue(self.tensor1 ==self.tensor1)
        self.assertTrue(self.tensor1 != self.tensor2)


if __name__ == "__main__":
    unittest.main()
