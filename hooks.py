import torch, re
from torch import nn
from functools import wraps
from functools import partial, reduce
from torchinfo import summary
from rich.theme import Theme
from rich.console import Console
from rich.traceback import install

from layers import LayerInfo
from model_info import ModelInfo

install(show_locals=True, theme="monokai", word_wrap=True)


# Magic function to get torch model layers by name using dot notations
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return reduce(_getattr, [obj] + attr.split("."))


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
    "name",
    "depth",
    "index",
    "parent",
    "is_leaf",
    "children",
    "trainable",
    "class_name",
    "input_shape",
    "output_shape",
    "total_params",
    "trainable_params",
    "non_trainable_params",
]  # Add more fields as needed


class ModelHooks:
    """
    A class to manage hooks for PyTorch modules.
    """

    def __init__(self, model_info: ModelInfo, level: int | tuple = None):

        assert isinstance(
            level, (int, tuple)
        ), "Level must be an int or a tuple of two integers."
        self.level = (0, level) if isinstance(level, int) else level
        self.model_info = model_info
        self.hooks = []
        self.layer_info = []

    @property
    def included_children(self):
        return self.model_info.get_children(level=self.level)

    @property
    def included_gchildren(self):
        gchildren = {}
        for child_included in self.included_children:
            search_pattern = f"{child_included}.[^.]+$"
            _search_func = lambda x: re.match(search_pattern, x)
            gchildren[child_included] = list(
                filter(_search_func, self.model_info.module_list)
            )
        return gchildren

    @property
    def model(self):
        return self.model_info.model

    def get_parent_name(self, name):
        """
        Get the parent name of a given layer name.
        Args:
            name (str): The name of the layer.
        Returns:
            str: The parent name of the layer.
        """
        return (
            name.rsplit(".", maxsplit=1)[0]
            if "." in name
            else self.model.__class__.__name__
        )

    def layer_info_hook(self, module, input, output, name, children):
        class_name = module.__class__.__name__
        depth = len(name.split(".")) - 1
        parent = self.get_parent_name(name)
        input_shape = [tuple(i.shape) for i in input]
        layerinfo = LayerInfo(
            name=name,
            layer=module,
            depth=depth,
            parent=parent,
            children=children,
            class_name=class_name,
            input_shape=input_shape,
            output_shape=tuple(output.shape),
        )
        self.layer_info.append(layerinfo)

        print(
            f"{name:<15}{class_name:<10}{depth} I-Shape: {input_shape} O-Shape: {tuple(output.shape)}, children: {children}, parent: {parent}"
        )

    def register_layer_hooks(self, hook_fn):
        for layer in self.included_children:
            gchildren = self.included_gchildren.get(layer, "None")
            handle = rgetattr(self.model, layer).register_forward_hook(
                partial(hook_fn, name=layer, children=gchildren)
            )
            self.hooks.append(handle)

    def run(self, input_size):
        dummy_input = torch.randn(*input_size)
        # print("XXXX", dummy_input.shape)
        self.model(dummy_input)

    def remove_hooks(self):
        """
        Remove all hooks from the module.
        """
        for handle in self.hooks:
            handle.remove()


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

    mymodel = NestedModel()
    # mymodel = ImageMulticlassClassificationNet()
    n = 2
    mi = ModelInfo(mymodel)
    mh = ModelHooks(mi, level=n)

    print("model_check", mi.model is mh.model)
    print("non-train", mi.trainable_params)

    # mh.run((1, 8))
    print(len(mh.hooks))
    print(len(mh.layer_info))
    print("1 done")

    mh.register_layer_hooks(mh.layer_info_hook)
    print(len(mh.hooks))
    print(len(mh.layer_info))
    print("2 done")
    mh.run((1, 8))
    mh.remove_hooks()
    print(len(mh.layer_info))
    # print(len(mh.hooks))
    # print(len(m.layer_info))
    summary(
        mymodel,
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
