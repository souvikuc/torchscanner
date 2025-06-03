import timeit
import torch
import ctypes, re
from torch import nn
from torchinfo import summary
from torchvision import models
from rich import print as rprint
from collections import OrderedDict
from functools import cached_property
from itertools import accumulate, chain

from src.utils import rgetattr
from src.layer_info import LayerInfo, LayerName


# =======================================================================================
# class to extract and contain all possible information about a module (in pytorch)
# =======================================================================================
class ModelInfo:
    """
    A class to manage model information for PyTorch modules.
    """

    def __init__(
        self,
        model: nn.Module,
        level: int | tuple = 1,
        columns: list = None,
    ):
        self.model = model
        self.level = level
        self.ln = LayerName(root=self.model)

    @property
    def trainable(self):
        return any(p.requires_grad for p in self.model.parameters())

    @property
    def parameters(self):
        return [name for name in self.model.named_parameters()]

    @property
    def total_params(self) -> int:
        """
        Returns the total number of parameters in the model.
        Returns:
            The total number of parameters in the model.
        """
        return sum(p.numel() for p in self.model.parameters())

    @property
    def trainable_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @property
    def non_trainable_params(self):
        x = self.total_params - self.trainable_params
        # x = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        return x

    # =====================================================
    # forbidden methods to use outside
    # =====================================================
    def __non_leaf_fn(self, x):
        non_leaf_joining_fn = lambda x, y: ".".join([x, y])
        return list(accumulate(x.split(".")[:-1], non_leaf_joining_fn))

    def __get_model_depth(self, module: nn.Module) -> int:
        """
        Calculates the depth of a PyTorch model recursively.
        Args:
            model: The PyTorch model.
        Returns:
            The depth of the model.
        """
        max_depth = 0
        for n, child in module.named_children():
            if len(child._modules) != 0:
                d = 1 + self.__get_model_depth(child)
                max_depth = max(max_depth, d)
            else:
                max_depth = max(max_depth, 0)
        return max_depth

    def __get_duplicated_leaves(self, leaves: OrderedDict) -> OrderedDict:
        flipped = OrderedDict()
        for name, id_value in leaves.items():
            if id_value not in flipped:
                flipped[id_value] = [name]
            else:
                flipped[id_value].append(name)
        return flipped

    def __remove_lower_duplicated_leaves(self, leaves: OrderedDict) -> OrderedDict:
        unique_leaves = OrderedDict()
        d_leaves = self.__get_duplicated_leaves(leaves)

        _filter_func = lambda x: len(x.split("."))
        for id_value, names in d_leaves.items():
            top_level_name = max(names, key=_filter_func)
            unique_leaves[top_level_name] = id_value
        return unique_leaves

    def __get_leaves(
        self, model: nn.Module, root: str = "", result: dict = {}
    ) -> OrderedDict:

        for name, module in model._modules.items():
            if not module._modules:
                result[f"{root}.{name}"] = id(module)
            else:
                inner_names = self.__get_leaves(module, f"{root}.{name}", result)
                result.update(inner_names)
        return OrderedDict(result)

    def __construct_info(self, leaf_tuple: tuple) -> tuple:
        name, leaf = leaf_tuple
        children = self.__get_children(name)
        leaf_info = LayerInfo(name=name, layer=leaf, children=children, root=self.model)
        return self.ln.original_name(name), leaf_info

    def __get_children(self, name: str) -> OrderedDict:
        if self.ln.is_root(name):
            children_pattern = f"^[^.]+$"
        else:
            children_pattern = f"{name}.[^.]+$"

        children = {
            key: val
            for key, val in self.descendants.items()
            if re.search(children_pattern, key)
        }
        return OrderedDict(children)

    # =====================================================
    # accesible useful methods to use outside
    # =====================================================
    @cached_property
    def leaves(self):
        leaves = self.__get_leaves(self.model)
        unique_leaves = self.__remove_lower_duplicated_leaves(leaves)

        # Iterate over a copy of keys by using list() to avoid issues
        for key in list(unique_leaves.keys()):
            layer = ctypes.cast(unique_leaves.pop(key), ctypes.py_object).value
            name = self.ln.original_name(key)
            unique_leaves[name] = layer

        return unique_leaves

    @cached_property
    def non_leaves(self):
        non_leaves = map(self.__non_leaf_fn, list(self.leaves.keys()))
        non_leaves = chain(*non_leaves)
        non_leaves = OrderedDict.fromkeys(chain(non_leaves))
        for key in list(non_leaves.keys()):
            name = self.ln.original_name(key)
            non_leaves[name] = rgetattr(self.model, name)

        return non_leaves

    @property
    def descendants(self):

        return OrderedDict({**self.leaves, **self.non_leaves})

    @property
    def included_layers(self):
        _filter_func = lambda x: self.ln.depth(x[0]) <= self.level
        incl_layers = list(filter(_filter_func, list(self.descendants.items())))
        if len(list(incl_layers)) != 0:
            return OrderedDict(incl_layers)
        else:
            return OrderedDict({self.ln.root_name: self.model})

    @property
    def leaves_info(self):
        leaves_info = list(map(self.__construct_info, self.leaves.items()))
        return OrderedDict(leaves_info)

    @property
    def non_leaves_info(self):
        non_leaves_info = list(map(self.__construct_info, self.non_leaves.items()))
        return OrderedDict(non_leaves_info)

    @property
    def descendants_info(self):
        return OrderedDict({**self.leaves_info, **self.non_leaves_info})

    @property
    def included_layers_info(self):
        included_layers_info = list(
            map(self.__construct_info, self.included_layers.items())
        )
        return OrderedDict(included_layers_info)

    @cached_property
    def module_list(self) -> list:
        """
        Returns the names of the children of a PyTorch model.
        Args:
            module: The PyTorch model.
        Returns:
            A list of module names.
        """
        modules_all = map(lambda x: x[0], self.model.named_modules())
        return filter(len, modules_all)

    @cached_property
    def depth(self) -> int:
        """
        Calculates the depth of the model.
        Returns:
            The depth of the model.
        """
        return self.__get_model_depth(self.model)


if __name__ == "__main__":

    def main_func():

        model = NestedModel()

        n = 4
        mi = ModelInfo(model, n)

        rprint(mi.included_layers)

        # summary(
        #     model,
        #     (1, 3, 224, 224),
        #     depth=8,
        #     col_names=[
        #         "input_size",
        #         "output_size",
        #         # "num_params",
        #         # "params_percent",
        #     ],
        #     row_settings=["var_names", "depth"],
        # )

    duration = timeit.timeit(main_func, number=1)
    print(duration)
