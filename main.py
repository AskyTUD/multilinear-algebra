import multilinear_algebra as ma

if __name__ == "__main__":

    # test_val: dict[str, list[str]] = {"server": ["A", "V"]}

    tensor1 = ma.Tensor(type="^_", dimension=2, name="A")

    tensor2 = ma.Tensor()
    tensor2.initialize_tensor({"type": "__", "dimension": [2, 2], "name": "B"})

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
    tensor4.assign_values(values={(0, 0): 3, (0, 1): -3.5, (1, 0): 2.14, (1, 2): -0.3})
    print(tensor4)

    tensor5 = ma.Tensor(type="^_", dimension=2, name="A")
    tensor5.get_random_values()
    tensor5.idx('ab')


    print(ma.MLA.scalar(2))
