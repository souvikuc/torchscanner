import torch
from torch import nn
from torchinfo import summary
from functools import wraps, partial
from copy import deepcopy
from torchvision import models
from pprint import pprint


class ModelInfo:
    """
    A class to manage model information for PyTorch modules.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    @property
    def module_list(self) -> list:
        """
        Returns the names of the children of a PyTorch model.
        Args:
            module: The PyTorch model.
        Returns:
            A list of module names.
        """
        modules_all = map(lambda x: x[0], self.model.named_modules())
        return list(filter(len, modules_all))

    @property
    def depth(self) -> int:
        """
        Returns the depth of the model.
        Returns:
            The depth of the model.
        """
        modules_splitted = map(lambda x: x.split("."), self.module_list)
        modules_depth_index = map(len, modules_splitted)

        return max(modules_depth_index) - 1

    def get_children(self, level: tuple = None) -> list:
        if level[0] == level[1]:
            _child_func = lambda x: len(x.split(".")) == level[0] + 1
        else:
            mini, maxi = min(level), max(level)
            _child_func = lambda x: (len(x.split(".")) >= mini + 1) and (
                len(x.split(".")) <= maxi + 1
            )
        return list(filter(_child_func, self.module_list))

    # def get_model_depth(self, module: nn.Module) -> int:
    #     """
    #     Calculates the depth of a PyTorch model.
    #     Args:
    #         model: The PyTorch model.
    #     Returns:
    #         The depth of the model.
    #     """
    #     max_depth = 0
    #     for n, child in module.named_children():
    #         if len(child._modules) != 0:
    #             d = 1 + self.get_model_depth(child)
    #             max_depth = max(max_depth, d)
    #         else:
    #             max_depth = max(max_depth, 0)
    #     return max_depth

    # def is_leaf(self, module: nn.Module) -> bool:
    #     return len(module._modules) == 0

    # def separate_children(self, module: nn.Module) -> dict:
    #     """
    #     Separates the children of a PyTorch model into a dictionary.
    #     Args:
    #         model: The PyTorch model.
    #     Returns:
    #         A dictionary of the children of the model.
    #     """
    #     mono = dict(filter(lambda x: len(x[-1]._modules) == 0, module.named_children()))
    #     poly = dict(filter(lambda x: len(x[-1]._modules) != 0, module.named_children()))
    #     return dict([("mono", mono), ("poly", poly)])

    # def all_leaf(self, module: nn.Module) -> bool:
    #     return all(map(lambda x: self.is_leaf(x), module.children()))


if __name__ == "__main__":

    class ImageMulticlassClassificationNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 6, 3)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 3)
            self.flatten = nn.Flatten()
            # self.fc1 = nn.Linear(16 * 11 * 11, 128)  # out: (BS, 128)
            # self.fc2 = nn.Linear(128, 64)
            # self.fc3 = nn.Linear(64, NUM_CLASSES)
            self.relu = nn.ReLU()
            # self.softmax = nn.LogSoftmax()
            self.class_head1 = nn.Sequential(
                nn.Linear(16 * 11 * 11, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 5),
                nn.Softmax(dim=-1),
            )
            self.class_head2 = nn.Sequential(
                nn.Linear(16 * 11 * 11, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 5),
                nn.Softmax(dim=-1),
            )

        def forward(self, x):
            x = self.conv1(x)  # out: (BS, 6, 48, 48)
            x = self.relu(x)
            x = self.pool(x)  # out: (BS, 6, 24, 24)
            x = self.conv2(x)  # out: (BS, 16, 22, 22)
            x = self.relu(x)
            x = self.pool(x)  # out: (BS, 16, 11, 11)
            x = self.flatten(x)
            x1 = self.class_head1(x)  # out: (BS, NUM_CLASSES)
            x2 = self.class_head2(x)  # out: (BS, NUM_CLASSES)
            # x = self.fc1(x)
            # x = self.relu(x)
            # x = self.fc2(x)
            # x = self.relu(x)
            # x = self.fc3(x)
            # x = self.softmax(x)
            return x1, x2

    mymodel = ImageMulticlassClassificationNet()

    class Block(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.fc1 = nn.Linear(in_features, 10)
            # self.fc2 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(10, out_features)
            # self.relu = nn.ReLU()

        def forward(self, x):
            return self.fc2(self.fc1(x))

    # class NestedModel(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.block1 = Block(3, 16)
    #         self.block2 = Block(16, 32)
    #         self.nested_block1 = nn.Sequential(Block(32, 64), Block(64, 128))
    #         self.nested_block2 = nn.Sequential(Block(128, 256), Block(256, 512))
    #         self.nested_block = nn.Sequential(self.nested_block1, self.nested_block2)
    #         self.final_conv = nn.Conv2d(512, 10, kernel_size=1)

    #     def forward(self, x):
    #         x = self.block1(x)
    #         x = self.block2(x)
    #         x = self.nested_block(x)
    #         x = self.final_conv(x)
    #         return x

    class NestedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = Block(8, 16)
            self.block2 = Block(16, 32)
            self.nested_block = nn.Sequential(Block(32, 64), Block(64, 128))
            self.final_conv = nn.Linear(128, 10)

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.nested_block(x)
            x = self.final_conv(x)
            return x

    # model = Block(3, 16)
    model = NestedModel()
    # vgg = models.vgg19(weights=True)
    mi = ModelInfo(model)
    n = 1
    print(f"The depth of the model is: {mi.depth}")
    # print(f"The depth of the model is: {mi.get_depth}")
    print(f"The modules of the model is: {mi.module_list}")
    print(
        f"The {n}-th level children of the model are (w_acc):\n {mi.get_children(level=n)}",
        end="\n\n",
    )
    print(
        f"The {n}-th level children of the model are (wo_acc):\n {mi.get_children(level=n)}",
        end="\n\n\n\n\n",
    )

    summary(
        model,
        (1, 8),
        depth=n,
        col_names=[
            "input_size",
            "output_size",
            # "num_params",
            # "params_percent",
        ],
        row_settings=["var_names", "depth"],
    )
