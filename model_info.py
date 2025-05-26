import torch
from torch import nn
from torchinfo import summary
from functools import wraps, partial, reduce
from copy import deepcopy
from torchvision import models
from pprint import pprint


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return reduce(_getattr, [obj] + attr.split("."))


class ModelInfo:
    """
    A class to manage model information for PyTorch modules.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.output = []
        self.children = {}

    @property
    def depth(self) -> int:
        """
        Calculates the depth of the model.
        Returns:
            The depth of the model.
        """
        return self.get_model_depth(self.model)

    def get_model_depth(self, module: nn.Module) -> int:
        """
        Calculates the depth of a PyTorch model.
        Args:
            model: The PyTorch model.
        Returns:
            The depth of the model.
        """
        max_depth = 0
        for n, child in module.named_children():
            if len(child._modules) != 0:
                d = 1 + self.get_model_depth(child)
                max_depth = max(max_depth, d)
            else:
                max_depth = max(max_depth, 0)
        return max_depth

    def is_leaf(self, module: nn.Module) -> bool:
        return len(module._modules) == 0

    def separate_children(self, module: nn.Module) -> dict:
        """
        Separates the children of a PyTorch model into a dictionary.
        Args:
            model: The PyTorch model.
        Returns:
            A dictionary of the children of the model.
        """
        mono = dict(filter(lambda x: len(x[-1]._modules) == 0, module.named_children()))
        poly = dict(filter(lambda x: len(x[-1]._modules) != 0, module.named_children()))
        return dict([("mono", mono), ("poly", poly)])

    def all_leaf(self, module: nn.Module) -> bool:
        return all(map(lambda x: self.is_leaf(x), module.children()))

    def get_children(
        self,
        module: nn.Module,
        level: int = 0,
        index: int = 0,
        # result: dict = {},
        name: str = "base",
    ) -> dict:
        # module_depth = self.get_model_depth(module)
        # print(f"{'    '*index}entering index {index} level of {name,level}")
        # print(f"{'    '*index}child {list(dict(module.named_children()).keys())}")
        # if level > module_depth:
        #     level = module_depth
        result = {}

        if index == level:
            # print(f"{'    '*index}running base case at level {index,level} of {name}")
            # if (index == level) or (self.all_leaf(module)):
            result.update({name: dict(module.named_children())})
            # result[name] = dict(module.named_children())
            # return dict(module.named_children())
            print(result, end="\n\n")
            return result

        for i, (name_c, child) in enumerate(module.named_children()):
            print(
                # f"{'    '*index}(Before {name}.{name_c} at level {index} out of {level})"
            )
            if not self.is_leaf(child):
                x = self.get_children(child, level, index=index + 1, name=name_c)
                # print(f"{'    '*index}nl_XX", name_c, x, end="\n\n")
                # result[name_c] = x
                # print(f"{'    '*index}x{x}")
                result.update(x)
            else:
                # print(f"{'    '*index}l_XX", name_c, child, end="\n\n")
                # self.children[name_c] = child
                result.update({name_c: child})
                # result[name_c] = child
            # print(f"{'    '*index}so far", name_c, result, end="\n\n")

            # print(f"{'    '*index}after {index} level")
            # print(f"{'    '*index}processed {name_c}-----")
            # print(
            #     # f"{'    '*index}(After {name}.{name_c} at level {index} out of {level})",
            #     "\n",
            #     f"{'    '*index}{result}",
            #     end="\n\n",
            # )
            # print("    " * index, result)
            # e = {name: result}
            # print("    " * index, e, end="\n\n" * index)
        return result

    # function used for removing nested
    # lists in python using recursion
    def reemovNestings(self, l):
        for i in l:
            if type(i) == list:
                self.reemovNestings(i)
            else:
                self.output.append(i)

    def get_layer_names(self, level, model) -> list:
        """
        Returns the names of the layers in the model.
        Returns:
            A list of layer names.
        """
        # print(self.depth)
        try:
            assert level <= self.depth
        except:
            level = self.depth

        layer_names = []
        if level == 1:
            print("x")
            return [name for name, _ in model.named_children()]
        else:
            for name, child in model.named_children():
                if len(child._modules) != 0:
                    # layer_names.append(name)
                    self.reemovNestings(self.get_layer_names(level - 1, child))
                    # layer_names.extend(self.get_layer_names(child))
                    layer_names.append([f"{name}.{n}" for n in self.output])
                else:
                    layer_names.append([])

        return layer_names


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
    print(f"The depth of the model is: {mi.depth}")
    # print(f"The children of the model are:\n {mi.get_children(model,level=0)}")
    n = 2
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
    res = mi.get_children(model, level=n)
    pprint({k: v for k, v in res.items()})
