import torch
from torch import nn
from functools import wraps
from functools import partial

from layers import LayerInfo


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

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.layer_info = []

    def layer_info_hook(module, input, output, depth, index, parent):
        class_name = module.__class__.__name__
        # name = module.__name__
        # print(class_name)
        children = module._modules

        layerinfo = LayerInfo(
            # name=name,
            layer=module,
            depth=depth,
            index=index,
            parent=parent,
            children=children,
            class_name=class_name,
            input_shape=list(input[0].shape),
            output_shape=list(output.shape),
        )
        # result.append(
        #     [depth, index, class_name, list(input[0].shape), list(output.shape)]
        # )
        self.layer_info.append(layerinfo)

        print(
            f"{depth,index}  {class_name:<20} Input Shape: {str(input[0].shape):<30} Output Shape: {str(output.shape):<30}"
        )

    def add_register_hook_methods(self):
        @add_dynamic_method(self.model)
        def register_layer_hooks(module, hook_fn, depth, index=0, parent=None):

            while depth >= 0:
                for name, modl in module.named_children():
                    if not (len(modl._modules) == 0):
                        handle = register_layer_hooks(
                            modl, hook_fn, depth=depth - 1, index=0, parent=name
                        )
                        print("xx", name)
                        self.hooks.append(handle)

                    else:
                        print("yy", name)
                        handle = modl.register_forward_hook(
                            partial(
                                hook_fn, depth=depth, index=index + 1, parent=parent
                            )
                        )
                        # print("yy", handle)
                        self.hooks.append(handle)
                break

            # return self.hooks

    def run(self, input_size):
        dummy_input = torch.randn(*input_size)
        # print("XXXX", dummy_input.shape)
        self.model(dummy_input)

    def add_forward_hook(self, hook_fn):
        """
        Add a forward hook to the module.
        """
        handle = self.model.register_forward_hook(hook_fn)
        self.hooks.append(handle)
        return handle

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

    mymodel = ImageMulticlassClassificationNet()

    m = ModelHooks(mymodel)

    m.add_hook_methods()
    m.add_register_hook_methods()

    m.model.register_layer_hooks(m.model, m.model.layer_info_hook, 0)
    print(len(m.hooks))
    print(len(m.layer_info))

    m.run((1, 1, 50, 50))
    print(len(m.hooks))
    print(len(m.layer_info))
