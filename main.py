import numpy as np

import multilinear_algebra as ma

if __name__ == "__main__":

    # define scalar: version 1
    tensor1s = ma.Tensor(value=2, name="s")

    # define scalar: version 2
    tensor2s = ma.Tensor()
    tensor2s.initialize_tensor({"name": "s", "value": 3.2})

    # define scalar: version 3
    tensor3s = ma.Tensor()
    tensor3s.initialize_tensor({"name": "s"})
    tensor3s.assign_values(value={(): 2})

    # define tensor: version 1
    tensor1 = ma.Tensor(
        type="^_",
        dimension=2,
        name="A",
        value={(0, 0): 1.5, (0, 1): -1.2, (1, 0): 2.0, (1, 1): 3.5},
    )

    # define tensor: version 2
    tensor2 = ma.Tensor()
    tensor2.initialize_tensor(
        {
            "type": "^_",
            "dimension": [2, 2],
            "name": "B",
            "value": {(0, 0): 1.5, (0, 1): -1.2, (1, 0): 2.0, (1, 1): 3.5},
        }
    )

    # define tensor: version 3
    tensor3 = ma.Tensor()
    tensor3.initialize_tensor(
        {
            "type": "^_",
            "dimension": [2, 2],
            "name": "B",
        }
    )
    tensor3.assign_values(value={(0, 0): 1.5, (0, 1): -1.2, (1, 0): 2.0, (1, 1): 3.5})

    tensor12 = tensor1 + tensor2

    tensor3 = ma.Tensor()
    tensor3.initialize_tensor(
        {
            "type": "__",
            "dimension": [2, 2],
            "name": "C",
            "values": {(0, 0): 1, (0, 1): 1, (1, 0): 2, (1, 1): 3},
        }
    )

    tensor4 = ma.Tensor()

    tensor4.initialize_tensor(
        {
            "type": "__",
            "dimension": [2, 2],
            "name": "D",
        }
    )
    tensor4.assign_values(values={(0, 0): 3, (0, 1): -3.5, (1, 0): 2.14, (1, 1): -0.3})
    print(tensor4)

    tensor1 == tensor2

    tensor5 = ma.Tensor(type="^_", dimension=2, name="A")
    tensor5.get_random_values()
    tensor5.idx("ab")

    tensor1.index_order
    # print(ma.MLA.scalar(2))
