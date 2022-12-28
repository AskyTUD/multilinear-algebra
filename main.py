import numpy as np

import multilinear_algebra as ma

if __name__ == "__main__":

    # define scalar: version 1
    tensor1s = ma.Tensor(value=2, name="s")

    # define tensor: version 1
    tensor1 = ma.Tensor(
        type="^_",
        dimension=2,
        name="A",
        value={(0, 0): 1.5, (0, 1): -1.2, (1, 0): 2.0, (1, 1): 3.5},
    )
    tensor2 = ma.Tensor(
        type="^_",
        dimension=2,
        name="A",
        value={(0, 0): 1.1, (0, 1): -1.8, (1, 0): -1.5, (1, 1): 2.5},
    )

    tensor1.idx("ab")
    tensor2.idx("bc")
    bla = tensor2 * tensor1
    ma.Tensor.print_multiplication(tensor1, tensor2)

    bla.print_components()

    