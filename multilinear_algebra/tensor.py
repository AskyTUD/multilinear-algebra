""" tensor module to generate and calculate with tensors """
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


# import itertools as it
# from abc import ABCMeta, abstractmethod
# import numpy as np
# import warnings
# from tabulate import tabulate

import itertools as it
import random as rd
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from casadi import casadi as ca  # type: ignore
from tabulate import tabulate

from .efun import get_index_values, sort_ca_byList

Einstein = Tuple[List[str], List[str], str, List[str]]


class TensorBasic:
    """This class provides several methods to define and calculate with tensors"""

    def __init__(self, **kwargs: Any) -> None:

        self.dimension: list = []
        self.index_order: list = []
        self.indices: list = []
        self.name: str = ""
        self.name_components: str = ""
        self.type: tuple = ()
        self.value: dict = {}
        self.is_initialized: bool = False
        self.is_scalar: bool = False

        if kwargs:
            self.initialize_tensor(kwargs)

    def initialize_tensor(self, tensor_attributes: Dict[str, Any]) -> None:
        """_summary_

        Args:
            tensor_attributes (Dict[str, Any]): contains properties

        Raises:
            NameError: need to define a name!
            NameError: to initialize a tensor, use properties: type, dimension!
        """
        if not "name" in tensor_attributes.keys():
            raise NameError("need to define a name!")

        if len(tensor_attributes.keys()) == 1:
            self.is_scalar = True
            self.name = tensor_attributes["name"]
            self.is_initialized = True

        if "value" in tensor_attributes.keys():
            if not isinstance(tensor_attributes["value"], dict):
                self.is_scalar = True
                self.name = tensor_attributes["name"]
                self.value[()] = ca.DM(tensor_attributes["value"])
                self.is_initialized = True

        if not self.is_scalar:
            needed_keys = ["type", "dimension"]
            check = [bool(i_key in set(tensor_attributes.keys())) for i_key in needed_keys]
            if not all(check):
                raise NameError("to initialize a tensor, use properties: type, dimension!")

            if "indices" in tensor_attributes.keys():
                use_letterz = list(tensor_attributes["indices"])
            else:
                use_letterz = [chr(code) for code in range(945, 970)]

            if isinstance(tensor_attributes["dimension"], int):
                dimension_val = [
                    tensor_attributes["dimension"] for i_type in list(tensor_attributes["type"])
                ]
            else:
                dimension_val = tensor_attributes["dimension"]

            self.dimension = dimension_val

            index_order = list(tensor_attributes["type"])
            use_indices = use_letterz[0 : len(index_order)]
            type_indices = [i_type + use_letterz[ii] for ii, i_type in enumerate(index_order)]

            self.index_order = index_order
            self.indices = use_indices
            self.name = tensor_attributes["name"]
            self.name_components = tensor_attributes["name"] + "".join(type_indices)
            self.type = (index_order.count("^"), index_order.count("_"))
            self.is_initialized = True
            if "value" in tensor_attributes.keys():
                self.assign_values(value=tensor_attributes["value"])
            else:
                indices_tot = get_index_values(dimension_val[0], sum(self.type))
                self.value = {i_index: ca.DM(0) for i_index in indices_tot}

    def assign_values(self, value: Dict[tuple, float]) -> None:
        """assign to the tensor its values

        Args:
            values (dict): dict of values

        Raises:
            IndexError: tensor is not initialized -> no tensor indices are known!
            IndexError: index (x1, x2) is not an element of values!
        """
        if not self.is_initialized:
            raise IndexError("tensor is not initialized -> no tensor indices are known!")
        if self.is_scalar:
            self.value[()] = ca.DM(value[()])
        else:
            indices_tot = get_index_values(self.dimension[0], sum(self.type))
            for i_index in indices_tot:
                if i_index in value.keys():
                    self.value[i_index] = ca.DM(value[i_index])
                else:
                    raise IndexError("index " + str(i_index) + " is not an element of values!")

    def __repr__(self) -> str:
        return self.name_components

    def __str__(self) -> str:
        return self.name_components


class Tensor(TensorBasic):
    """define all operations of a class

    Args:
        Tensor (_type_): _description_
    """

    def rename(self: TensorBasic, new_name: str, n_t: bool = False) -> Union[TensorBasic, None]:
        """give the object a new name

        Args:
            self (TensorBasic): tensor
            new_name (str): new name of the tensor
            n_t (bool, optional): generate a new instance. Defaults to False.

        Returns:
            Union[TensorBasic, None]: new tensor
        """
        help_val = self.name_components.split(self.name)[1]
        self.name = new_name
        self.name_components = new_name + help_val

        if n_t:
            return deepcopy(self)
        return None

    def idx(self: TensorBasic, new_index: str, n_t: bool = False) -> Union[TensorBasic, None]:
        """give the object new indices

        Args:
            self (TensorBasic): tensor
            new_index (str): new indices
            n_t (bool, optional): generate a new instance. Defaults to False.

        Raises:
            IndexError: mismatch in the number of indices!

        Returns:
            Union[TensorBasic, None]: new tensor
        """
        if len(self.index_order) != len(new_index):
            raise IndexError("mismatch in the number of indices!")
        type_indices = [i_type + new_index[ii] for ii, i_type in enumerate(self.index_order)]
        self.indices = list(new_index)
        self.name_components = self.name + "".join(type_indices)

        if n_t:
            return deepcopy(self)
        return None

    def get_random_values(
        self: TensorBasic, lower_bound: int = -10, upper_bound: int = 10, mode: str = "general"
    ) -> None:
        """get random numbers to initialize tensors

        Args:
            lower_bound (int, optional): lower bound for the random generator. Defaults to 10.
            upper_bound (int, optional): upper bound for the random gnerator. Defaults to 10.
            type (str, optional): option between general and quadratic form. Defaults to "general".

        Raises:
            TypeError: _description_
        """
        if mode == "general":
            indices_tot = get_index_values(self.dimension[0], len(self.indices))
            for i_index in indices_tot:
                self.value[i_index] = ca.DM(rd.randint(lower_bound, upper_bound))
        if mode == "quadratic_form":
            if "".join(self.index_order) == "__":
                indices_tot = get_index_values(self.dimension[0], len(self.indices))
                for i_index in indices_tot:
                    val = ca.DM(rd.randint(1, upper_bound))
                    self.value[i_index] = val
                    self.value[tuple(reversed(i_index))] = val
            else:
                raise TypeError("No quadratic form; use type=general!")

    def val(self: TensorBasic) -> Union[float, None]:
        """returns the scalar value

        Raises:
            TypeError: xxx is not a scalar!

        Returns:
            _type_: value of a scalar
        """
        if self.is_scalar:
            return self.value[()].full()
        raise TypeError(self.name + " is not a scalar!")

    def print_components(self: TensorBasic) -> None:
        """list all the individual components in a table

        Args:
            self (TensorBasic): tensor

        Returns:
            _type_: table of components
        """
        if self.is_scalar:
            tab_raw = [["", self.name, str(self.value[()])]]
        else:
            index_values = list(it.product(range(self.dimension[0]), repeat=len(self.indices)))

            def index_fun(index_list: list) -> list:
                return [val + str(index_list[i]) for i, val in enumerate(self.index_order)]

            # index_fun = lambda x: [val + str(x[i]) for i, val in enumerate(self.index_order)]
            tab_raw = []
            for i_index in index_values:
                help_flag = [
                    str(i_index),
                    self.name + "".join(index_fun(list(i_index))),
                    self.value[i_index].str(),
                ]
                tab_raw.append(help_flag)

        print(tabulate(tab_raw, headers=["Index", "Symbol", "Value"]))

    def __eq__(self, other) -> bool:
        """compare two tensors if they are identical

        Args:
            other (_type_): _description_

        Returns:
            bool: _description_
        """
        if all(self.dimension) == all(other.dimension) and self.type == other.type:
            indices_tot = get_index_values(self.dimension[0], len(self.indices))
            bool_comp = [
                abs(self.value[i_index] - other.value[i_index]) < 1e-9 for i_index in indices_tot
            ]
            return bool(all(bool_comp))
        return False

    def __neg__(self: TensorBasic) -> TensorBasic:
        """get the negative MLA object by inverting the sign of each component

        Args:
            self (TensorBasic): tensor

        Returns:
            TensorBasic: tensor with negative value
        """
        if self.is_scalar:
            return Tensor(name="(-" + self.name + ")", value=-self.value[()])

        new_tensor = Tensor(
            type="".join(self.index_order),
            name="(-" + self.name + ")",
            dimension=self.dimension,
        )
        for key, value in self.value.items():
            new_tensor.value[key] = -value
        return new_tensor

    def __add__(self: TensorBasic, other: TensorBasic) -> Union[TensorBasic, None]:
        """calculate the sum of two tensors

        Args:
            self (TensorBasic): first tensor
            other (TensorBasic): second tensor

        Raises:
            TypeError: invalid addition of scalar and tensor!
            TypeError: both tensors are not in the same space!
            TypeError: indices of the two tensors do not match!

        Returns:
            Union[TensorBasic, None]: new tensor
        """

        if self.is_scalar != other.is_scalar:
            raise TypeError("invalid addition of scalar and tensor!")

        if self.is_scalar and other.is_scalar:
            return Tensor(
                name="(" + self.name + "+" + other.name + ")",
                value=self.value[()] + other.value[()],
            )

        if not self.is_scalar and not other.is_scalar:
            if not Tensor.is_same_space(self, other):
                raise TypeError("both tensors are not in the same space!")
            if not Tensor.is_valid_indices(self, other, mode="addition"):
                raise TypeError("indices of the two tensors do not match!")

            new_tensor = Tensor(
                type="".join(self.index_order),
                name="(" + self.name + "+" + other.name + ")",
                dimension=self.dimension,
            )
            for key, value in self.value.items():
                new_tensor.value[key] = value + other.value[key]
            return new_tensor
        return None

    def __sub__(self: TensorBasic, other: TensorBasic) -> Union[TensorBasic, None]:
        """calculate the difference of two tensors

        Args:
            self (TensorBasic): first tensor
            other (TensorBasic): second tensor

        Returns:
            TensorBasic: new tensor
        """
        return self + other.__neg__()

    def __mul__(self: TensorBasic, other: TensorBasic) -> Union[TensorBasic, None]:
        """multiplication of two tensors

        Args:
            self (TensorBasic): first tensor
            other (TensorBasic): second tensor

        Raises:
            TypeError: indices are not suitable for multiplication!

        Returns:
            Union[TensorBasic, None]: product of the two tensors
        """

        if self.is_scalar and other.is_scalar:
            return Tensor(
                name="(" + self.name + other.name + ")", value=self.value[()] * other.value[()]
            )

        if self.is_scalar and not other.is_scalar:
            new_tensor = Tensor(
                type="".join(other.index_order),
                name="(" + self.name + other.name + ")",
                dimension=other.dimension[0],
                indices="".join(other.indices),
            )
            for key, value in other.value.items():
                new_tensor.value[key] = self.value[()] * value
            return new_tensor

        if not self.is_scalar and other.is_scalar:
            new_tensor = Tensor(
                type="".join(self.index_order),
                name="(" + other.name + self.name + ")",
                dimension=self.dimension[0],
                indices="".join(self.indices),
            )
            for key, value in self.value.items():
                new_tensor.value[key] = other.value[()] * value
            return new_tensor

        if not self.is_scalar and not other.is_scalar:
            if Tensor.is_valid_indices(self, other, mode="multiplication"):
                sum_over, free_indices, free_indices_order, _ = Tensor.get_indice_list(self, other)
                new_tensor = Tensor(
                    type=free_indices_order,
                    name="(" + self.name + "*" + other.name + ")",
                    dimension=self.dimension[0],
                    indices="".join(free_indices),
                )

                get_ten1_idx, get_ten2_idx = Tensor.get_index_projection(
                    self, other, free_indices, sum_over
                )

                new_indices_val = get_index_values(self.dimension[0], len(free_indices))
                sum_indices_val = get_index_values(self.dimension[0], len(sum_over))
                for i_index in new_indices_val:
                    new_val_help = 0
                    for i_sum_index in sum_indices_val:
                        a_ind = tuple(
                            map(int, np.array(get_ten1_idx(i_index, i_sum_index)).tolist()[0])
                        )
                        b_ind = tuple(
                            map(int, np.array(get_ten2_idx(i_index, i_sum_index)).tolist()[0])
                        )
                        new_val_help += self.value[a_ind] * other.value[b_ind]
                    new_tensor.value[i_index] = ca.DM(new_val_help)
                return new_tensor
            raise TypeError("indices are not suitable for multiplication!")
        return None

    @staticmethod
    def print_multiplication(tensor1: TensorBasic, tensor2: TensorBasic) -> None:
        """print the full multiplication table

        Args:
            tensor1 (TensorBasic): first tensor
            tensor2 (TensorBasic): second tensor

        Raises:
            TypeError: index mismatch; Einstein summation rule can not be applied!

        Returns:
            _type_: _description_
        """
        if tensor1.is_scalar and tensor2.is_scalar:
            print("not yet supported")

        if tensor1.is_scalar and not tensor2.is_scalar:
            print("not yet supported")

        if not tensor1.is_scalar and tensor2.is_scalar:
            print("not yet supported")

        if not tensor1.is_scalar and not tensor2.is_scalar:
            if Tensor.is_valid_indices(tensor1, tensor2, mode="multiplication"):
                sum_over, free_indices, free_indices_order, _ = Tensor.get_indice_list(
                    tensor1, tensor2
                )

                index_values_result = get_index_values(tensor1.dimension[0], len(free_indices))

                def index_fun_result(idx_list: list) -> list:
                    return [val + str(idx_list[i]) for i, val in enumerate(free_indices_order)]

                name_result = "(" + tensor1.name + "*" + tensor2.name + ")"
                index_values_sum_over = get_index_values(tensor1.dimension[0], len(sum_over))

                def index_fun_ten(idx_list: list, tensor: TensorBasic) -> list:
                    return [val + str(idx_list[i]) for i, val in enumerate(tensor.index_order)]

                get_ten1_idx, get_ten2_idx = Tensor.get_index_projection(
                    tensor1, tensor2, free_indices, sum_over
                )

                tab_raw = []
                for i_index in index_values_result:
                    help_sum = ""
                    for i_sum_index in index_values_sum_over:
                        a_ind = tuple(
                            map(int, np.array(get_ten1_idx(i_index, i_sum_index)).tolist()[0])
                        )
                        b_ind = tuple(
                            map(int, np.array(get_ten2_idx(i_index, i_sum_index)).tolist()[0])
                        )
                        ten1_ind = tensor1.name + "".join(index_fun_ten(list(a_ind), tensor1))
                        ten2_ind = tensor2.name + "".join(index_fun_ten(list(b_ind), tensor2))
                        help_sum += ten1_ind + " " + ten2_ind + " + "
                    help_flag = [
                        str(i_index),
                        name_result + "".join(index_fun_result(list(i_index))),
                        help_sum[:-2],
                    ]
                    tab_raw.append(help_flag)
                print(tabulate(tab_raw, headers=["Index", "Symbol", "Value"]))
            else:
                raise TypeError("index mismatch; Einstein summation rule can not be applied!")

    @staticmethod
    def is_same_space(tensor1: TensorBasic, tensor2: TensorBasic) -> bool:
        """check if both tensors are of same dimension and type

        Args:
            tensor1 (TensorBasic): first tensor
            tensor2 (TensorBasic): second tensor

        Returns:
            bool: are both tensors lie in the same space
        """
        return tensor1.dimension == tensor2.dimension and tensor1.type == tensor2.type

    @staticmethod
    def is_valid_indices(tensor1: TensorBasic, tensor2: TensorBasic, mode: str) -> bool:
        """check if the Einstein rule for multiplication can be applied

        Args:
            tensor1 (TensorBasic): first tensor
            tensor2 (TensorBasic): second tensor
            mode (str): addition | multiplication

        Returns:
            bool: are the indices valid for addition/multiplication
        """
        if mode == "addition":
            # indices of both terms must match
            return bool(
                tensor1.index_order == tensor2.index_order and tensor1.indices == tensor2.indices
            )
        if mode == "multiplication":
            # same indices must be of different types
            _, _, _, unset_indices = Tensor.get_indice_list(tensor1, tensor2)
            if unset_indices:
                return False
            return True
        return False

    @staticmethod
    def get_indice_list(tensor1: TensorBasic, tensor2: TensorBasic) -> Einstein:
        """check for free indices and those to be used for summation

        Args:
            tensor1 (TensorBasic): first tensor
            tensor2 (TensorBasic): second tensor

        Returns:
            Einstein: tuple of indice information
        """
        sum_over = list(set(tensor1.indices).intersection(tensor2.indices))
        tensor1_idx_pos = []
        tensor2_idx_pos = []
        unset_indices = []
        if sum_over:
            for name in sum_over:
                ten1_help = tensor1.indices.index(name)
                ten2_help = tensor2.indices.index(name)
                if tensor1.index_order[ten1_help] != tensor2.index_order[ten2_help]:
                    tensor1_idx_pos.append(ten1_help)
                    tensor2_idx_pos.append(ten2_help)
                else:
                    unset_indices.append(name)

        new_types_tensor1 = [
            ix for ii, ix in enumerate(tensor1.index_order) if ii not in tensor1_idx_pos
        ]
        new_types_tensor2 = [
            ix for ii, ix in enumerate(tensor2.index_order) if ii not in tensor2_idx_pos
        ]
        free_indices_order = "".join(new_types_tensor1 + new_types_tensor2)
        free_indices = [ix for ix in tensor1.indices + tensor2.indices if ix not in sum_over]

        return sum_over, free_indices, free_indices_order, unset_indices

    @staticmethod
    def get_index_projection(
        tensor1: TensorBasic, tensor2: TensorBasic, free_indices: List[str], sum_over: List[str]
    ) -> Tuple[ca.Function, ca.Function]:
        """generate projection maps for multiplication

        Args:
            tensor1 (TensorBasic): first tensor
            tensor2 (TensorBasic): second tensor
            free_indices (List[str]): indices of the new tensor
            sum_over (List[str]): indices used for summation

        Returns:
            Tuple[ca.Function, ca.Function]: projection maps for both tensors
        """
        # generate indices for summation
        tot_indices = free_indices + sum_over
        ind_letter = [ca.SX.sym(i_name) for i_name in tot_indices]
        # assign these indices to the elements
        new_ind_letter = ca.hcat(
            sort_ca_byList(
                [i_letter for i_letter in ind_letter if i_letter.name() in free_indices],
                free_indices,
            )
        )
        sum_ind_letter = ca.hcat(
            sort_ca_byList(
                [i_letter for i_letter in ind_letter if i_letter.name() in sum_over], sum_over
            )
        )
        a_ind_letter = ca.hcat(
            sort_ca_byList(
                [i_letter for i_letter in ind_letter if i_letter.name() in tensor1.indices],
                tensor1.indices,
            )
        )
        b_ind_letter = ca.hcat(
            sort_ca_byList(
                [i_letter for i_letter in ind_letter if i_letter.name() in tensor2.indices],
                tensor2.indices,
            )
        )
        # generate function to get the indices
        get_ten1_idx = ca.Function(
            "get_a_index",
            [new_ind_letter, sum_ind_letter],
            [a_ind_letter],
            ["freeIndices", "sumOverIndices"],
            ["indicesOfA"],
        )
        get_ten2_idx = ca.Function(
            "get_b_index",
            [new_ind_letter, sum_ind_letter],
            [b_ind_letter],
            ["freeIndices", "sumOverIndices"],
            ["indicesOfB"],
        )

        return get_ten1_idx, get_ten2_idx
