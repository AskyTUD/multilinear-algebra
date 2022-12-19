import multilinear_algebra as ma

if __name__ == "__main__":
    n_dim = 2
    s = ma.MLA.scalar(2)
    u = ma.MLA.parameter("u")
    v = ma.MLA.parameter("v")

    A = ma.MLA(tensor_type=["^_"], name="A", dim=n_dim)
    A.get_random_values()
    print("the components of A are given by: ", A)
    A.print_components()
