import torch
from torch import nn
from torchinfo import summary
from torchvision import models
from rich.tree import Tree
from rich import print as rprint
from bigtree import dict_to_tree

import timeit
from src.hooks import ModelHooks
from src.model_info import ModelInfo


class TorchTree:
    """
    A class to manage a tree structure of PyTorch modules.
    """

    def __init__(
        self,
        model_hooks: ModelHooks,
    ):
        self.model_hooks = model_hooks

    @property
    def model(self):
        return self.model_hooks.model

    def get_dummy_inputs(self, input_data=None, input_size=None):
        """
        Returns a dummy input tensor based on the model's input shape.
        Returns:
            A dummy input tensor.
        """
        input_data_specified = (input_data is not None) or (input_size is not None)

        if input_data_specified and input_size is not None:
            return torch.randn(*input_size)
        elif input_data_specified and input_data is not None:
            return input_data
        else:
            raise ValueError("Input data or size must be defined in the model.")


if __name__ == "__main__":

    def main_func():

        mymodel = NestedModel()
        n = 0
        summary_table(mymodel, input_size=(1, 3, 512, 512), level=n)
        summary_tree(mymodel, input_size=(1, 3), level=n)

        # summary(
        #     mymodel,
        #     (1, 3),
        #     depth=n,
        #     col_names=[
        #         "input_size",
        #         "output_size",
        #         # "num_params",
        #         # "params_percent",
        #     ],
        #     row_settings=["var_names", "depth"],
        # )

    duration = timeit.timeit(main_func, number=100)
    print(duration)
