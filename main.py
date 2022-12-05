
import multilinear_algebra as ma

if __name__ == '__main__':

    n_dim = 2
    u_prim = ma.MLA.parameter('u')
    a_prim = ma.MLA.parameter('a')
    ap = ma.MLA.scalar(a_prim)
    sp = ma.MLA.scalar(u_prim)

    A = ma.MLA(tensor_type=['^_'], name='A', dim=2)
    A.get_random_values()
    P = ma.MLA(tensor_type=['__'], name='P', dim=2)
    P.get_random_values(type='quadratic_form')
    P.values[(0, 0)] = ap.val()
    C = P.id('ab') * A.id('bd')
    Pn = sp*C

    M = A + A

    Pn(u=1, a=3)
    Pn.print_components()