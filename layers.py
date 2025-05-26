import torch
from torch import nn
from functools import wraps


class LayerInfo:
    def __init__(
        self,
        name,
        layer,
        depth,
        # index,
        parent,
        children,
        class_name,
        input_shape,
        output_shape,
    ):
        self.name = name
        self.layer = layer
        self.depth = depth
        # self.index = index
        self.parent = parent
        self.children = children
        self.class_name = class_name
        self.input_shape = input_shape
        self.output_shape = output_shape
        # self.children = []

    @property
    def is_leaf(self):
        return len(self.layer._modules) == 0

    @property
    def trainable(self):
        return any(p.requires_grad for p in self.layer.parameters())

    @property
    def total_params(self):
        return sum(p.numel() for p in self.layer.parameters())

    @property
    def non_trainable_params(self):
        return sum(p.numel() for p in self.layer.parameters() if not p.requires_grad)

    def __repr__(self):
        return f"{self.class_name}"
        # return f"{self.class_name} : {self.layer}"


class Layers:
    """
    A class to manage layers for PyTorch modules.
    """

    def __init__(self, module: nn.Module, depth: int = None):
        self.module = module
        self.layers = []
        self.depth = depth

    def add_layer(self, layer_fn):
        """
        Add a layer to the module.
        """
        self.layers.append(layer_fn)
        return layer_fn

    def remove_layers(self):
        """
        Remove all layers from the module.
        """
        for layer in self.layers:
            # Assuming each layer has a remove method
            layer.remove()  # This is a placeholder; actual implementation may vary
