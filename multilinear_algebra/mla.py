#
#   This file is part of multilinear_algebra
#
#   multilinear_algebra is a package providing
#   methods for addition, multiplication, representation, etc., of multilinear objects
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

    def __call__(self, *args, **kwargs):
        """
        allows to evaluate MLA object in case the components contain symbolic primitives
        :param args:
        :param kwargs:
        :return:
        """
        key_list = []
        components = []
        for key, value in self.values.items():
            key_list.append(key)
            components.append(value)
        prim_list = ef.getPrimitives(components)
        prim_list_name = [i_prim.name() for i_prim in prim_list]
        prim_list_val = [kwargs[i_name] for i_name in prim_list_name]
        comp_fun = ca.Function('evalComponents', prim_list, components)
        comp_val = comp_fun.call(prim_list_val)
        for id, key in enumerate(key_list):
            self.values[key] = comp_val[id]

    def __eq__(self, other):
        """
        compare if two MLA objects are identical
        :param other:
        :return:
        """
        if all(self.dimension) == all(other.dimension) and self.type == other.type:
            indices_val = MLA.get_index_values(self.dimension[0], np.sum(self.type))
            bool_comp = [abs(self.values[i_index] - other.values[i_index]) < 1e-9 for i_index in indices_val]
            if all(bool_comp):
                return True
            else:
                return False
        else:
            return False

    def __neg__(self):
        """
        get the negative MLA object by inverting the sign of each component
        :return:
        """
        if self.is_scalar():
            new_tensor = MLA.scalar(-self.values[()])
        else:
            new_tensor = MLA(tensor_type=[''.join(self.index_order)], name='(-' + self.name + ')', dim=self.dimension[0])
            for key, value in self.values.items():
                new_tensor.values[key] = -value

        return new_tensor

    def __add__(self, other):
        """
        add two MLA objects
        :param other:
        :return:
        """
        flag_a, mode_a, info_a = MLA.check_type(self, other)  #MLA.check4compatibility(self, other)
        if mode_a == 1:
            flag_b, mode_b, info_b = MLA.check_dimension(self, other)
            if mode_b == 1:
                new_tensor = MLA.scalar(self.values[()] + other.values[()])
            if mode_b == 4:
                flag_c = MLA.is_addition_valid(self, other)
                if flag_c:
                    new_tensor = MLA(tensor_type=[''.join(self.index_order)],
                                     name='(' + self.name + '+' + other.name + ')',
                                     dim=self.dimension[0])
                    for key, value in self.values.items():
                        new_tensor.values[key] = value + other.values[key]
                else:
                    raise TypeError('inconsistency of the indices of the MLA objects')
            if mode_b not in {1, 4}:
                raise TypeError('inconsistency in the dimension of the MLA objects')
        else:
            raise TypeError('inconsistency of MLA object types')

        return new_tensor

    def __sub__(self, other):
        """
        subtract two MLA objects
        :param other:
        :return:
        """
        flag_a, mode_a, info_a = MLA.check_type(self, other)
        if mode_a == 1:
            flag_b, mode_b, info_b = MLA.check_dimension(self, other)
            if mode_b == 1:
                new_tensor = MLA.scalar(self.values[()] - other.values[()])
            if mode_b == 4:
                flag_c = MLA.is_addition_valid(self, other)
                if flag_c:
                    new_tensor = MLA(tensor_type=[''.join(self.index_order)],
                                     name='(' + self.name + '-' + other.name + ')',
                                     dim=self.dimension[0])
                    for key, value in self.values.items():
                        new_tensor.values[key] = value - other.values[key]
                else:
                    raise TypeError('inconsistency of the indices of the MLA objects')
            if mode_b not in {1, 4}:
                raise TypeError('inconsistency in the dimension of the MLA objects')
        else:
            raise TypeError('inconsistency of MLA object types')

        return new_tensor

    def __mul__(self, other):
        """
        multiply two MLA objects -> contraction or increasing the indices
        :param other:
        :return:
        """
        flag_a, mode_a, info_a = MLA.check_dimension(self, other)
        if mode_a == 1:
            new_tensor = MLA.scalar(self.values[()] * other.values[()])
        if mode_a == 2:
            new_tensor = MLA(tensor_type=[''.join(other.index_order)],
                             name='(' + self.name + '*' + other.name + ')',
                             dim=other.dimension[0])
            for key, value in other.values.items():
                new_tensor.values[key] = self.values[()] * value
        if mode_a == 3:
            new_tensor = MLA(tensor_type=[''.join(self.index_order)],
                             name='(' + other.name + '*' + self.name + ')',
                             dim=self.dimension[0])
            for key, value in self.values.items():
                new_tensor.values[key] = other.values[()] * value
        if mode_a == 4:
            n_dim = self.dimension[0]

            flagE, sumOverIndices, freeIndices, freeIndicesOrder, type = MLA.is_Einstein_valid(self, other)

            if flagE:
                # generate raw object
                new_tensor = MLA(tensor_type=[freeIndicesOrder],
                                 name='(' + self.name + '*' + other.name + ')',
                                 dim=n_dim,
                                 letterz=''.join(freeIndices))

                get_a_index, get_b_index = MLA.get_index_projection(self, other, freeIndices, sumOverIndices)

                new_indices_val = MLA.get_index_values(n_dim, len(freeIndices))
                sum_indices_val = MLA.get_index_values(n_dim, len(sumOverIndices))
                for i_index in new_indices_val:
                    new_val_help = 0
                    for i_sum_index in sum_indices_val:
                        a_ind = tuple(map(int, get_a_index(i_index, i_sum_index).full().tolist()[0]))
                        b_ind = tuple(map(int, get_b_index(i_index, i_sum_index).full().tolist()[0]))
                        new_val_help += self.values[a_ind] * other.values[b_ind]
                    new_tensor.values[i_index] = new_val_help
            else:
                raise TypeError('index mismatch; Einstein summation rule can not be applied!')
        if not flag_a:
            raise TypeError('inconsistency in the dimension of the MLA objects')

        return new_tensor

    def rename(self, str):
        """
        change the name of the MLA object
        :param str:
        :return:
        """
        help = self.name_components.split(self.name)[1]
        self.name = str
        self.name_components = str + help

    def id(self, str):
        """
        change/specify the indices of the MLA object -> import when multiply two objects
        :param str:
        :return:
        """
        new_tensor = MLA(tensor_type=[''.join(self.index_order)], name=self.name, dim=self.dimension[0], letterz=str)
        new_tensor.values = self.values
        return new_tensor

    def val(self):
        """
        returns the scalar value
        :return:
        """
        if self.is_scalar():
            return self.values[()]
        else:
             raise TypeError(self.name + ' is not a scalar!')

    def get_random_values(self, lb=-10, ub=10, type='general'):
        """
        initialize an MLA object with random integer numbers
        :param lb:
        :param ub:
        :param type:
        :return:
        """
        if type == 'general':
            index_values = MLA.get_index_values(self.dimension[0], len(self.indices))
            for i_index in index_values:
                self.values[i_index] = ca.DM(rd.randint(lb, ub))
        if type == 'quadratic_form':
            if ''.join(self.index_order) == '__':
                index_values = MLA.get_index_values(self.dimension[0], len(self.indices))
                for i_index in index_values:
                    val = ca.DM(rd.randint(1, ub))
                    self.values[i_index] = val
                    self.values[tuple(reversed(i_index))] = val
                    # index_values.remove(tuple(reversed(i_index)))

    def get_matrix(self):
        """
        transform the MLA object into a classical matrix or vector
        :return:
        """
        N = len(self.indices)
        if N == 1:
            flag_symbolic = False
            A0 = ca.SX(np.zeros(self.dimension))
            if self.index_order[0] == '_':
                A0 = A0.T
            for ir in range(self.dimension[0]):
                val = self.values[(ir,)]
                if isinstance(val, ca.SX):
                    flag_symbolic = True
                A0[ir] = val
        if N == 2:
            flag_symbolic = False
            A0 = ca.SX(np.zeros(self.dimension))
            for ir in range(self.dimension[0]):
                for ic in range(self.dimension[1]):
                    val = self.values[(ir, ic)]
                    if isinstance(val, ca.SX):
                        flag_symbolic = True
                    A0[ir, ic] = val
        if N > 2:
            out = None
            warnings.WarningMessage('noRuleToBuildAMatrix')
        out = A0 if flag_symbolic else ca.DM(A0)

        return out

    def T(self):
        """
        transpose the MLA object
        :return:
        """
        N = len(self.indices)
        if N == 1:
            new_type = ['_'] if self.index_order[0] == '^' else ['^']
            new_tensor = MLA(tensor_type=new_type, name='(' + self.name + 'T)', dim=self.dimension[0])
            for key, value in self.values.items():
                new_tensor.values[key] = value
        if N == 2:
            new_tensor = MLA(tensor_type=[''.join(self.index_order[::-1])], name='(' + self.name + 'T)', dim=self.dimension[0])
            for key, value in self.values.items():
                new_tensor.values[key] = self.values[tuple(reversed(key))]
        if N > 2:
            out = None
            warnings.WarningMessage('noRule4Transpose')

        return new_tensor

    def print_components(self):
        """
        list all the individual components
        :return:
        """
        if self.is_scalar():
            tab_raw = [['', self.name, str(self.values[()])]]
        else:
            index_values = list(it.product(range(self.dimension[0]), repeat=len(self.indices)))
            index_fun = lambda x: [val + str(x[i]) for i, val in enumerate(self.index_order)]
            tab_raw = []
            for i_index in index_values:
                help = [str(i_index), self.name + ''.join(index_fun(list(i_index))), self.values[i_index].str()]
                tab_raw.append(help)

        print(tabulate(tab_raw, headers=['Index', 'Symbol', 'Value']))

    def is_scalar(self):
        """
        check if the MLA object is a scalar
        :return: boolean
        """
        return True if self.type == (0, 0) else False

    def equal_dimension(self):
        """
        check if all dimensions of the indices match
        :return:
        """
        return all(ele == self.dimension[0] for ele in self.dimension)

    @staticmethod
    def print_multiplication(self, other):
        """
        print the full multiplication table
        :param A:
        :param B:
        :return:
        """
        # flag, mode, info = MLA.check4compatibility(A, B)
        flag_a, mode_a, info_a = MLA.check_dimension(self, other)
        if mode_a == 1:
            print('not yet supported')
        if mode_a == 2:
            print('not yet supported')
        if mode_a == 3:
            print('not yet supported')
        if mode_a == 4:
            n_dim = self.dimension[0]
            flagE, sumOverIndices, freeIndices, freeIndicesOrder, type = MLA.is_Einstein_valid(self, other)

            if flagE:
                index_values_result = MLA.get_index_values(n_dim, len(freeIndices))
                index_fun_result = lambda x: [val + str(x[i]) for i, val in enumerate(freeIndicesOrder)]
                name_result = '(' + self.name + '*' + other.name + ')'

                index_values_sumOver = MLA.get_index_values(n_dim, len(sumOverIndices))
                index_fun_A = lambda x: [val + str(x[i]) for i, val in enumerate(self.index_order)]
                index_fun_B = lambda x: [val + str(x[i]) for i, val in enumerate(other.index_order)]

                get_a_index, get_b_index = MLA.get_index_projection(self, other, freeIndices, sumOverIndices)
                tab_raw = []
                for i_index in index_values_result:
                    help_sum = ''
                    for i_sum_index in index_values_sumOver:
                        a_ind = tuple(map(int, get_a_index(i_index, i_sum_index).full().tolist()[0]))
                        b_ind = tuple(map(int, get_b_index(i_index, i_sum_index).full().tolist()[0]))
                        A_ind = self.name + ''.join(index_fun_A(list(a_ind)))
                        B_ind = other.name + ''.join(index_fun_B(list(b_ind)))
                        help_sum += A_ind + ' ' + B_ind + '+'
                    help = [str(i_index), name_result + ''.join(index_fun_result(list(i_index))), help_sum[:-1]]
                    tab_raw.append(help)
                print(tabulate(tab_raw, headers=['Index', 'Symbol', 'Value']))
            else:
                raise TypeError('index mismatch; Einstein summation rule can not be applied!')
        if not flag_a:
            raise TypeError('inconsistency in the dimension of the MLA objects')

    @ staticmethod
    def get_index_values(n_dim, n_indices):
        """
        get the indices of the MLA object
        :param n_dim:
        :param n_indices:
        :return:
        """
        return list(it.product(range(n_dim), repeat=n_indices))

    @ staticmethod
    def scalar(val, name='s'):
        """
        define a scalar MLA object
        :param val:
        :param name:
        :return:
        """
        new_tensor = MLA(tensor_type=[''], name=name, dim=0)
        new_tensor.values = {(): val}
        return new_tensor

    @ staticmethod
    def parameter(name):
        """
        define a symbolic parameter for an MLA object
        :param name:
        :return:
        """
        return ca.SX.sym(name)

    @ staticmethod
    def get_Kronecker(tensor_type=[''], dim=2):
        """
        get a Kronecker MLA object
        :param tensor_type:
        :param dim:
        :return:
        """
        new_tensor = MLA(tensor_type=tensor_type, name=chr(948), dim=dim)
        indeces_val = MLA.get_index_values(dim, 2)
        for i_index in indeces_val:
            new_tensor.values[i_index] = ca.DM(1) if i_index[0] == i_index[1] else ca.DM(0)
        return new_tensor

    @ staticmethod
    def check_dimension(self, other):
        """
        check if both objects have the same dimensions
        :param self:
        :param other:
        :return:
        """
        if self.is_scalar() and other.is_scalar():
            return True, 1, 'operateWith2Scalars'
        if self.is_scalar():
            return True, 2, 'selfIsScalar'
        if other.is_scalar():
            return True, 3, 'otherIsScalar'
        if self.equal_dimension() and other.equal_dimension() and self.dimension[0] == other.dimension[0]:
            return True, 4, 'matchingDimension'
        else:
            raise TypeError('dimension mismatch')

    @ staticmethod
    def check_type(self, other):
        """
        check if both objects have the same type
        :param self:
        :param other:
        :return:
        """
        if self.type == other.type:
            return True, 1, '4mult_add'
        else:
            return True, 2, '4mult'

    @staticmethod
    def is_addition_valid(self, other):
        """
        check if the summation rule for addition can be applied
        :param self:
        :param other:
        :return:
        """
        if self.index_order == other.index_order and self.indices == other.indices:
            return True
        else:
            return False

    @ staticmethod
    def is_Einstein_valid(self, other):
        """
        check if the Einstein rule for multiplication can be applied
        :param self:
        :param other:
        :return:
        """
        sumOverIndices = list(set(self.indices).intersection(other.indices))
        a_index_pos = []
        b_index_pos = []
        if sumOverIndices:
            for name in sumOverIndices:
                a_help = self.indices.index(name)
                b_help = other.indices.index(name)
                if self.index_order[a_help] != other.index_order[b_help]:
                    a_index_pos.append(a_help); b_index_pos.append(b_help)
                else:
                    return False, [], [], [], 'IndexMismatch;identicalIndexTypesAreFound'
            type = 'contraction'
        else:
            type = 'extension'

        new_types_fromA = [ix for ii, ix in enumerate(self.index_order) if ii not in a_index_pos]
        new_types_fromB = [ix for ii, ix in enumerate(other.index_order) if ii not in b_index_pos]
        residualIndicesOrder = ''.join(new_types_fromA + new_types_fromB)
        residualIndices = [ix for ix in self.indices + other.indices if ix not in sumOverIndices]

        return True, sumOverIndices, residualIndices, residualIndicesOrder, type

    @ staticmethod
    def get_index_projection(self, other, freeIndices, sumOverIndices):
        """
        get function to map from a list of free and sumover indices to the one of the MLA objects
        :param self:
        :param other:
        :param freeIndices:
        :param sumOverIndices:
        :return:
        """

        # generate indices for summation
        tot_indices = freeIndices + sumOverIndices
        ind_letter = [ca.SX.sym(i_name) for i_name in tot_indices]
        # assign these indices to the elements
        new_ind_letter = ca.hcat(ef.sort_ca_byList([i_letter for i_letter in ind_letter if i_letter.name() in freeIndices], freeIndices))
        sum_ind_letter = ca.hcat(ef.sort_ca_byList([i_letter for i_letter in ind_letter if i_letter.name() in sumOverIndices], sumOverIndices))
        a_ind_letter = ca.hcat(ef.sort_ca_byList([i_letter for i_letter in ind_letter if i_letter.name() in self.indices], self.indices))
        b_ind_letter = ca.hcat(ef.sort_ca_byList([i_letter for i_letter in ind_letter if i_letter.name() in other.indices], other.indices))
        # generate function to get the indices
        get_a_index = ca.Function('get_a_index', [new_ind_letter, sum_ind_letter], [a_ind_letter],
                                  ['freeIndices', 'sumOverIndices'], ['indicesOfA'])
        get_b_index = ca.Function('get_b_index', [new_ind_letter, sum_ind_letter], [b_ind_letter],
                                  ['freeIndices', 'sumOverIndices'], ['indicesOfB'])

        return get_a_index, get_b_index