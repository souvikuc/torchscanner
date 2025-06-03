import torch
from torch import nn

# from torchinfo import summary
from torchvision import models
from rich.tree import Tree
from rich import print as rprint
from bigtree import dict_to_tree
from functools import cached_property
import timeit

from src.hooks import ModelHooks
from src.model_info import ModelInfo
from src.enums import LayerInfoSettings

from sample_models_test import *


def summary_table(
    model: nn.Module,
    input_data=None,
    input_size=None,
    level: int | tuple = None,
    columns: list = None,
):
    model_info = ModelInfo(model, level, columns)
    model_hooks = ModelHooks(model_info)
    model_hooks.register_layer_hooks(model_hooks.layer_info_hook)
    # torchtree = TorchTree(model_hooks)

    # dummy_inputs = torchtree.get_dummy_inputs(input_data, input_size)
    # torchtree.model(dummy_inputs)
    model_hooks.run(input_size)
    model_hooks.remove_hooks()


def summary_tree(
    model: nn.Module, input_data=None, input_size=None, level: int | tuple = None
):
    model_info = ModelInfo(model, level, columns=None)
    root = model_info.ln.root_name
    tree_dict = {}
    for name, layer_info in model_info.included_layers_info.items():
        full_name = model_info.ln.full_name(name)
        is_tr = " *" if layer_info.infodict[LayerInfoSettings.TRAINABLE] else ""
        tree_dict[full_name] = {
            LayerInfoSettings.CLASSNAME.value: layer_info.infodict[
                LayerInfoSettings.CLASSNAME
            ]
            + is_tr
        }
    tree = dict_to_tree(tree_dict, sep=".")
    tree.show(attr_list=[LayerInfoSettings.CLASSNAME.value], attr_bracket=("(", ")"))


if __name__ == "__main__":

    def main_func():

        mymodel = Model_1()
        n = 4
        summary_table(mymodel, input_size=(1, 3), level=n, columns=None)
        summary_tree(mymodel, input_size=(1, 3), level=n)

        # summary(
        #     mymodel,
        #     (1, 3, 224, 224),
        #     depth=n,
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
