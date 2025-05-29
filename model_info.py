import torch
from torch import nn
from torchinfo import summary
from torchvision import models

from layer_info import LayerInfo


class ModelInfo:
    """
    A class to manage model information for PyTorch modules.
    """

    def __init__(self, model: nn.Module):
        self.model = model

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
        return filter(len, modules_all)

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
    # mymodel = models.vgg19(weights=True)

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
    # model = NestedModel()
    model = models.vgg19(weights=True)
    mi = ModelInfo(model)
    n = 1

    # summary(
    #     model,
    #     (1, 3, 512, 512),
    #     depth=n,
    #     col_names=[
    #         "input_size",
    #         "output_size",
    #         # "num_params",
    #         # "params_percent",
    #     ],
    #     row_settings=["var_names", "depth"],
    # )
