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

import itertools as it

import numpy as np
from casadi import casadi as ca


def get_index_values(n_dim, n_indices):
    """
    get the indices of the MLA object
    :param n_dim:
    :param n_indices:
    :return:
    """
    return list(it.product(range(n_dim), repeat=n_indices))


def size_casadi_vec(list):
    """
    get the length of a casadi variable/ an empty list
    :param list:
    :return:
    """
    if isinstance(list, ca.SX):
        return list.size()[0]
    else:
        return 0


def flatten(list):
    return [ie for ilist in list for ie in ilist]


def powerset(s):
    """
    calculation of the powerset
    :param s: list
    :return: list of all sublists
    """
    x = len(s)
    out = []
    for i in range(1 << x):
        element = [s[j] for j in range(x) if (i & (1 << j))]
        out.append(element)
    return out


def truncate(P, A, direction='right'):
    """
    cutting the sequence P at a specific value if it is also an element in A
    :param P: sequence (tuple) of numbers
    :param A: set of numbers
    :param direction: specifies if we cut from the left (A|P) or from the right (P|A)
    :return: a truncated sequence or the original ones
    """
    P_ = P.copy()
    if direction == 'right':
        P_out = []
        for ip in P_:
            P_out.append(ip)
            if ip in A:
                return P_out

    if direction == 'left':
        P_out = []
        P_.reverse()
        for ip in P_:
            P_out.append(ip)
            if ip in A:
                P_out.reverse()
                return P_out


def getColor(name):
    """
    return the HEX of a color
    :param name:
    :return: colorcode
    """
    if name == 'CCPSblue1':
        return '#254061'
    if name == 'CCPSblue2':
        return '#386192'
    if name == 'CCPSturquoise1':
        return '#31859c'
    if name == 'CCPSgray2':
        return '#bfbfbf'
    if name == 'CCPSred2':
        return '#953735'
    if name == 'CCPSgreen1':
        return '#43503a'
    if name == 'CCPSgreen2':
        return '#aaab80'
    if name == 'CCPSsand':
        return '#ece9d4'
    if name == 'CCPStext':
        return '#383C3C'


def getListFromDictList(listOfDict, identifier, order, subOrder=None):
    """
    The function gets from a list of dictionaries those values which key match to the identifier.
    If the value of this key is a list, only the entries specified by the subOrder are selected.
    :param listOfDict: list
    :param identifier: string
    :param order: list of integers
    :param subOrder: list of entries specified in the order key
    :return: list of values
    """
    OUT = []
    if subOrder:
        subIdentifier = identifier[0] + '_order'

        for idx, inode in enumerate(order):
            pr = getProjection(listOfDict[inode-1].get(subIdentifier), subOrder[idx])
            help = pr(listOfDict[inode - 1].get(identifier)).full().transpose().tolist()[0]
            OUT = OUT + help
    else:
        for inode in order:
            if not isinstance(listOfDict[inode-1].get(identifier), list):
                help = [listOfDict[inode-1].get(identifier)]
            else:
                help = listOfDict[inode-1].get(identifier)
            OUT = OUT + help
    return OUT


def getProjection(dom, tar):
    """
    The function returns a casadi projection map.
    :param dom: list
    :param tar: list
    :return:
    """
    dom_prim = ca.SX.sym('dom', dom.__len__())
    tar_prim = []
    for idx, i_dom in enumerate(dom):
       if i_dom in tar:
           tar_prim = ca.vertcat(tar_prim, dom_prim[idx])
    return ca.Function('projection', [dom_prim], [tar_prim])


def getNames(ca_prim):
    """
    extract from a casadi primitiv the names of the individual variables
    :param ca_prim: casadi primitiv
    :return: ca_str: list of strings representing the names of the primitives
    """
    if isinstance(ca_prim, ca.SX):
        NameList = [ielement.name() for ielement in ca_prim.elements()]
        return ', '.join(NameList)
    else:
        return ''


def getPrimitives(ca_exp):
    """
    get from list of primitives from a casadi expression
    :param ca_exp:
    :return:
    """
    help_prime = ca.Function('primitiv', [], ca_exp)
    return help_prime.free_sx()


def getExpression(ca_fun, *argv):
    """
    extract from a casadi function the primitives and expressions
    :param ca_fun: casadi function
    :param argv: list of strings to name the primitives
    :return: dict_of_primitives, dict_of_expressions
    """
    if len(argv) < 1:
        name = ['p'+str(i_in) for i_in in range(ca_fun.n_in())]
    else:
        name = [argv[0][i_in] if i_in <= len(argv[0]) else 'p'+str(i_in) for i_in in range(ca_fun.n_in())]
    primitives_name = {ca_fun.name_in(i_in): ca.SX.sym(name[i_in], ca_fun.nnz_in(i_in)) for i_in in range(ca_fun.n_in())}
    return primitives_name, ca_fun.call(primitives_name)


def getItem(dictList, identifier, value):
    """
    returns the dictionary of a list where the identifier as a specific value
    :param dictList:
    :param identifier:
    :param value:
    :return: dictionary
    """
    return next((item for item in dictList if item[identifier] == value), None)


def str2tup(string):
    return eval(string.split('_')[1])


def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)


def n_l(ca_list):
    if str(type(ca_list)) == "<class 'casadi.casadi.SX'>":
        return ca_list.size1()
    else:
        return 0


def get_permutation(list):
    sigma = [list.index(x) for x in sorted(list)]
    PM = np.zeros([len(sigma), len(sigma)])
    for ir in range(len(sigma)):
        for ic in range(len(sigma)):
            if sigma[ir] == ic:
                PM[ir, ic] = 1
    X = ca.SX.sym('X', len(sigma))
    return ca.Function('permutation', [X], [np.matrix(PM)@X])


def sort_ca_byList(list1, list2):
    """
    returns a list of the casadi primitives in list1 but ordered by the names in list2
    :param list1:
    :param list2:
    :return:
    """
    list1_name = [i_name.name() for i_name in list1]
    return [list1[list1_name.index(i_element)] for i_element in list2]
