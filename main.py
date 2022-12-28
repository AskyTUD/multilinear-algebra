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

    tensor1.idx("bc")
    tensor2.idx("ab")
    bla = tensor2 * tensor1

    bla.print_components()

    tensor2.idx("ab")
    tensor1.idx("bc")
    tensor12 = tensor2 * tensor1
    bla = tensor1s + tensor1s
