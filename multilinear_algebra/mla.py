#
#   This file is part of multilinear_algebra
#
#   multilinear_algebra is package providing methods for addition, multiplication, representation, etc., of multilinear objects
#
#   Copyright (c) 2022 Andreas Himmel
#                      All rights reserved
#
#   multilinear_algebra is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   multilinear_algebra is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU Lesser General Public License
#   along with multilinear_algebra. If not, see <http://www.gnu.org/licenses/>.
#

import warnings
import random as rd
import itertools as it
import numpy as np
from tabulate import tabulate
from casadi import casadi as ca
import multilinear_algebra.efun as ef


class MLA:
    """
    CLASS FOR MULTILINEAR ALGEBRA

    Notes
    ------
    -

    """

    def __init__(self, tensor_type=[''], name='', dim=2, letterz=[]):
        """
        generate a multilinear object
        :param tensor_type: list containing a string -> ^ contravariant or _ covariant
        :param name: name of the object
        :param dim: dimension of the underlying space
        :param letterz: list of a string -> index letters
        """
        greek_letterz = [chr(code) for code in range(945, 970)]
        latin_latterz = [chr(code) for code in range(97, 122)]
        l2g = lambda a: greek_letterz[latin_latterz.index(a)]
        use_letterz = list(letterz) if letterz else greek_letterz

        shape_val = [dim for i in tensor_type[0]] if isinstance(dim, int) else dim
        index_order = [i for i in tensor_type[0]]
        indices = use_letterz[0:len(index_order)]
        type_indices = [i_type + use_letterz[ii] for ii, i_type in enumerate(index_order)]
        self.dimension = shape_val
        self.index_order = index_order
        self.indices = indices    # ''.join(indices)
        self.name = name
        self.name_components = name + ''.join(type_indices)
        self.type = (index_order.count('^'), index_order.count('_'))
        indices_tot = MLA.get_index_values(dim, sum(self.type))
        self.values = {i_index: ca.DM(0) for i_index in indices_tot}  # np.zeros(shape=shape_val)

    def __repr__(self):
        return str(self.name_components)

    def __str__(self):
        return self.name_components

    @ staticmethod
    def get_index_values(n_dim, n_indices):
        return list(it.product(range(n_dim), repeat=n_indices))
