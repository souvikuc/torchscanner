import torch
from torch import nn


class LayerInfo:
    def __init__(
        self,
        name,
        layer,
        depth,
        parent,
        children,
        class_name,
        input_shape,
        output_shape,
    ):
        self.name = name
        self.layer = layer
        self.depth = depth
        self.parent = parent
        self.children = children
        self.class_name = class_name
        self.input_shape = input_shape
        self.output_shape = output_shape

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
    def trainable_params(self):
        return sum(p.numel() for p in self.layer.parameters() if p.requires_grad)

    @property
    def non_trainable_params(self):
        x = self.total_params - self.trainable_params
        # x = sum(p.numel() for p in self.layer.parameters() if not p.requires_grad)
        return x

    def __repr__(self):
        return f"{self.class_name}"
        # return f"{self.class_name} : {self.layer}"
