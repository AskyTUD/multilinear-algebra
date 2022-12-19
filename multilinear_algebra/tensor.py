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

import random as rd

# import numpy as np
from casadi import casadi as ca

import multilinear_algebra.efun as ef

# import typing


# import warnings


# from tabulate import tabulate


class Tensor:
    """This class provides several methods to define and calculate with tensors"""

    def __init__(self, **kwargs) -> None:

        self.dimension: list = []
        self.index_order: list = []
        self.indices: list = []
        self.name: str = ""
        self.name_components: str = ""
        self.type: tuple = ()
        self.values: dict = {}

        if kwargs:
            self.initialize_tensor(kwargs)

    def initialize_tensor(self, tensor_attributes: dict) -> None:
        """init the name, type and so on to the class attributes

        Args:
            tensor_attributes (dict): _description_

        Raises:
            Exception: _description_
        """
        needed_keys = ["type", "name", "dimension"]
        check = [
            True if i_key in set(tensor_attributes.keys()) else False
            for i_key in needed_keys
        ]
        if not all(check):
            raise Exception("need to define: type, name, dimension!")

        if "indices" in tensor_attributes.keys():
            use_letterz = list(tensor_attributes["indices"])
        else:
            use_letterz = [chr(code) for code in range(945, 970)]

        if isinstance(tensor_attributes["dimension"], int):
            dimension_val = [
                tensor_attributes["dimension"]
                for i_type in list(tensor_attributes["type"])
            ]
        else:
            dimension_val = tensor_attributes["dimension"]

        self.dimension = dimension_val

        index_order = list(tensor_attributes["type"])
        use_indices = use_letterz[0 : len(index_order)]
        type_indices = [
            i_type + use_letterz[ii] for ii, i_type in enumerate(index_order)
        ]

        self.index_order = index_order
        self.indices = use_indices
        self.name = tensor_attributes["name"]
        self.name_components = tensor_attributes["name"] + "".join(type_indices)
        self.type = (index_order.count("^"), index_order.count("_"))
        if "values" in tensor_attributes.keys():
            self.assign_values(tensor_attributes["values"])
        else:
            indices_tot = ef.get_index_values(dimension_val[0], sum(self.type))
            self.values = {i_index: ca.DM(0) for i_index in indices_tot}

    def assign_values(self, values: dict) -> None:
        """assign to the tensor its values
        Args:
            values (dict): _description_
        Raises:
            TypeError: _description_
        """
        indices_tot = ef.get_index_values(self.dimension[0], sum(self.type))
        for i_index in indices_tot:
            if i_index in values.keys():
                self.values[i_index] = ca.DM(values[i_index])
            else:
                raise TypeError(
                    "Index " + str(i_index) + " is not an element of values!"
                )

    def __repr__(self) -> str:
        return self.name_components

    def __str__(self) -> str:
        return self.name_components

    def rename(self, new_name: str) -> None:
        """give the object a new name

        Args:
            new_name (str): _description_
        """
        help_val = self.name_components.split(self.name)[1]
        self.name = new_name
        self.name_components = new_name + help_val

    def idx(self, new_index: str) -> None:
        """give the object new indices

        Args:
            new_index (str): _description_

        Raises:
            IndexError: _description_
        """
        if len(self.index_order) != len(new_index):
            raise IndexError("mismatch in the number of indices!")
        type_indices = [
            i_type + new_index[ii] for ii, i_type in enumerate(self.index_order)
        ]
        self.indices = list(new_index)
        self.name_components = self.name + "".join(type_indices)

    def get_random_values(
        self, lower_bound: int = -10, upper_bound: int = 10, type: str = "general"
    ) -> None:
        """get random numbers to initialize tensors

        Args:
            lower_bound (int, optional): _description_. Defaults to 10.
            upper_bound (int, optional): _description_. Defaults to 10.
            type (str, optional): _description_. Defaults to "general".

        Raises:
            TypeError: _description_
        """
        if type == "general":
            indices_tot = ef.get_index_values(self.dimension[0], len(self.indices))
            for i_index in indices_tot:
                self.values[i_index] = ca.DM(rd.randint(lower_bound, upper_bound))
        if type == "quadratic_form":
            if "".join(self.index_order) == "__":
                indices_tot = ef.get_index_values(self.dimension[0], len(self.indices))
                for i_index in indices_tot:
                    val = ca.DM(rd.randint(1, upper_bound))
                    self.values[i_index] = val
                    self.values[tuple(reversed(i_index))] = val
            else:
                raise TypeError("No quadratic form; use type=general!")
