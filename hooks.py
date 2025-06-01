import torch, re
import timeit
from torch import nn
from functools import wraps
from functools import partial, reduce
from torchinfo import summary
from torchvision import models

from rich import print as rprint

from layer_info import LayerInfo, LayerName
from model_info import ModelInfo
from utils import rgetattr


def add_dynamic_method(self):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):  # def wrapper(self, *args, **kwargs):
            return func(*args, **kwargs)

        setattr(self, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func  # returning func means func can still be used normally

    return decorator


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


class ModelHooks:
    """
    A class to manage hooks for PyTorch modules.
    """

    def __init__(self, model_info: ModelInfo):
        # def __init__(self, model_info: ModelInfo, level: int | tuple = None):
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
        layerinfo = LayerInfo(
            name=name,
            layer=module,
            children=children,
            input_shape=input_shape,
            output_shape=tuple(output.shape),
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

    # class ImageMulticlassClassificationNet(nn.Module):
    #     def __init__(self) -> None:
    #         super().__init__()
    #         self.conv1 = nn.Conv2d(1, 6, 3)
    #         self.pool = nn.MaxPool2d(2, 2)
    #         self.conv2 = nn.Conv2d(6, 16, 3)
    #         self.flatten = nn.Flatten()
    #         # self.fc1 = nn.Linear(16 * 11 * 11, 128)  # out: (BS, 128)
    #         # self.fc2 = nn.Linear(128, 64)
    #         # self.fc3 = nn.Linear(64, NUM_CLASSES)
    #         self.relu = nn.ReLU()
    #         # self.softmax = nn.LogSoftmax()
    #         self.class_head1 = nn.Sequential(
    #             nn.Linear(16 * 11 * 11, 128),
    #             nn.ReLU(),
    #             nn.Linear(128, 64),
    #             nn.ReLU(),
    #             nn.Linear(64, 5),
    #             nn.Softmax(dim=-1),
    #         )
    #         self.class_head2 = nn.Sequential(
    #             nn.Linear(16 * 11 * 11, 128),
    #             nn.ReLU(),
    #             nn.Linear(128, 64),
    #             nn.ReLU(),
    #             nn.Linear(64, 5),
    #             nn.Softmax(dim=-1),
    #         )

    #     def forward(self, x):
    #         x = self.conv1(x)  # out: (BS, 6, 48, 48)
    #         x = self.relu(x)
    #         x = self.pool(x)  # out: (BS, 6, 24, 24)
    #         x = self.conv2(x)  # out: (BS, 16, 22, 22)
    #         x = self.relu(x)
    #         x = self.pool(x)  # out: (BS, 16, 11, 11)
    #         x = self.flatten(x)
    #         x1 = self.class_head1(x)  # out: (BS, NUM_CLASSES)
    #         x2 = self.class_head2(x)  # out: (BS, NUM_CLASSES)
    #         # x = self.fc1(x)
    #         # x = self.relu(x)
    #         # x = self.fc2(x)
    #         # x = self.relu(x)
    #         # x = self.fc3(x)
    #         # x = self.softmax(x)
    #         return x1, x2

    # mymodel = ImageMulticlassClassificationNet()

    # mymodel = models.vgg19(weights=True)
    def main_func():
        class Block(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.fc1 = nn.Linear(in_features, 10)
                # self.fc2 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(10, out_features)
                # self.relu = nn.ReLU()

            def forward(self, x):
                return self.fc2(self.fc1(x))

        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block1 = Block(3, 16)
                self.block2 = Block(16, 32)
                self.nested_block1 = nn.Sequential(Block(32, 64), Block(64, 128))
                self.nested_block2 = nn.Sequential(Block(128, 256), Block(256, 512))
                self.nested_block = nn.Sequential(
                    self.nested_block1, self.nested_block2
                )
                # self.nested_block = nn.Sequential(
                #     nn.Sequential(Block(32, 64), Block(64, 128)),
                #     nn.Sequential(Block(128, 256), Block(256, 512)),
                # )

            def forward(self, x):
                x = self.block1(x)
                x = self.block2(x)
                x = self.nested_block(x)
                # x = self.final_conv(x)
                return x

        # model = Block(3, 16)
        # model = NestedModel()
        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        # rprint(model)
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
