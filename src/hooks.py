import timeit
import torch, re
from torch import nn
from functools import wraps
from torchinfo import summary
from functools import partial
from torchvision import models

from rich import print as rprint

from src.utils import rgetattr
from src.model_info import ModelInfo
from src.layer_info import LayerInfo, LayerName


info_fields = [
    "original_name",
    "depth",
    # "is_leaf",
    "children",
    # "trainable",
    "class_name",
    "input_shape",
    "output_shape",
    # "total_params",
    # "trainable_params",
    # "non_trainable_params",
]  # Add more fields as needed


# =======================================================================================
# class to create and attach and remove hooks to a set of layers/modules (in pytorch)
# =======================================================================================
class ModelHooks:
    """
    A class to manage hooks for PyTorch modules.
    """

    def __init__(self, model_info: ModelInfo):
        self.model_info = model_info
        self.hooks = []
        self.layer_info = []

    @property
    def included_children(self):
        return self.model_info.included_layers

    @property
    def included_children_info(self):
        return self.model_info.included_layers_info

    @property
    def model(self):
        return self.model_info.model

    def layer_info_hook(self, module, input, output, name, children):
        input_shape = [tuple(i.shape) for i in input]
        # print("xxxx", output[0])
        output_shape = []
        if isinstance(output, tuple):
            for i in output:
                if isinstance(i, tuple):
                    for j in i:
                        output_shape.append(tuple(j.shape))
                        # rprint(j)
                else:
                    output_shape.append(tuple(i.shape))
                    # rprint(i)
        else:
            output_shape.append(output.shape)
        # print("yyyy", input)
        layerinfo = LayerInfo(
            name=name,
            layer=module,
            children=children,
            input_shape=input_shape,
            output_shape=output_shape,
            root=self.model,
        )
        self.layer_info.append(layerinfo)
        rprint(layerinfo.infodict)

    def register_layer_hooks(self, hook_fn):
        for name, layer_info in self.included_children_info.items():
            gchildren = layer_info.children
            if name == self.model_info.ln.root_name:
                handle = self.model.register_forward_hook(
                    partial(hook_fn, name=name, children=gchildren)
                )
            else:
                handle = rgetattr(self.model, name).register_forward_hook(
                    partial(hook_fn, name=name, children=gchildren)
                )
            self.hooks.append(handle)

    def remove_hooks(self):
        """
        Remove all hooks from the module.
        """
        for handle in self.hooks:
            handle.remove()

    def run(self, shape):
        inp = torch.rand(shape)
        self.model(inp)


if __name__ == "__main__":

    # mymodel = models.vgg19(weights=True)
    def main_func():

        model = NestedModel()
        n = 2
        mi = ModelInfo(model, n)
        mh = ModelHooks(mi)

        mh.register_layer_hooks(mh.layer_info_hook)
        mh.run((1, 3, 224, 224))
        mh.remove_hooks()
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
