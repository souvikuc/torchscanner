import torch
from torch import nn
from torchinfo import summary
from torchvision import models
from rich.tree import Tree
from rich import print as rprint
from bigtree import dict_to_tree
from functools import cached_property
import timeit

from src.hooks import ModelHooks
from src.model_info import ModelInfo
from src.enums import LayerInfoSettings


def summary_table(
    model: nn.Module, input_data=None, input_size=None, level: int | tuple = None
):
    model_info = ModelInfo(model, level)
    model_hooks = ModelHooks(model_info)
    model_hooks.register_layer_hooks(model_hooks.layer_info_hook)
    # torchtree = TorchTree(model_hooks)

    # dummy_inputs = torchtree.get_dummy_inputs(input_data, input_size)
    # torchtree.model(dummy_inputs)
    model_hooks.run((1, 3))
    model_hooks.remove_hooks()


def summary_tree(
    model: nn.Module, input_data=None, input_size=None, level: int | tuple = None
):
    model_info = ModelInfo(model, level)
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
                # self.nested_block1 = nn.Sequential(Block(32, 64), Block(64, 128))
                # self.nested_block2 = nn.Sequential(Block(128, 256), Block(256, 512))
                # self.nested_block = nn.Sequential(
                #     self.nested_block1, self.nested_block2
                # )
                self.nested_block = nn.Sequential(
                    nn.Sequential(Block(32, 64), Block(64, 128)),
                    nn.Sequential(Block(128, 256), Block(256, 512)),
                )
                # self.final_conv = nn.Conv2d(512, 10, kernel_size=1)

            def forward(self, x):
                x = self.block1(x)
                x = self.block2(x)
                x = self.nested_block(x)
                # x = self.final_conv(x)
                return x

        mymodel = NestedModel()
        # mymodel = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        # mymodel = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # mymodel = ImageMulticlassClassificationNet()
        n = 4
        summary_table(mymodel, input_size=(1, 3), level=n)
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
