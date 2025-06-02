import torch
from torch import nn

from src.enums import LayerInfoSettings


# =======================================================================================
# class to manipulate layer names in dot notation (like pytorch)
# =======================================================================================
class LayerName:

    def __init__(self, root):
        self.root_name = root.__class__.__name__

    def full_name(self, name):
        if self.is_full_name(name):
            return name
        else:
            delemeter = "" if name.startswith(".") else "."
            return f"{self.root_name}{delemeter}{name}"

    def original_name(self, name):
        if self.is_full_name(name):
            x = name.split(".", maxsplit=1)
            original_name = x[-1] if len(x) == 2 else self.root_name
            return original_name
        else:
            return name

    def is_root(self, name):
        return name == self.root_name

    def is_full_name(self, name):
        return name.startswith(".") or name.startswith(self.root_name)

    def depth(self, name):
        n = 1 if self.is_full_name(name) else 0
        return len(name.split(".")) - n

    def parent(self, name):
        x = name.rsplit(".", maxsplit=1)
        parent = x[0] if len(x) == 2 else self.root_name
        return parent

    def basename(self, name):
        x = name.rsplit(".", maxsplit=1)
        if self.is_full_name(name):
            basename = x[-1] if len(x) == 2 else self.root_name
        else:
            basename = x[-1]

        return basename


# =======================================================================================
# class to contain all possible information about a layer/module (in pytorch)
# =======================================================================================


class LayerInfo:
    def __init__(
        self,
        name=None,
        layer=None,
        children=None,
        input_shape=None,
        output_shape=None,
        root=None,
    ):
        self.info_dict = {}
        self.layer = layer
        self.children = children
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.class_name = layer.__class__.__name__

        self.ln = LayerName(root)
        self.depth = self.ln.depth(name)
        self.parent = self.ln.parent(name)
        self.basename = self.ln.basename(name)
        self.full_name = self.ln.full_name(name)
        self.original_name = self.ln.original_name(name)

    @property
    def is_leaf(self):
        return len(self.layer._modules) == 0

    @property
    def parameters(self):
        return [name for name, param in self.layer.named_parameters()]

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

    @property
    def infodict(self):
        # info_dict = {}
        self.info_dict[LayerInfoSettings.DEPTH] = self.depth
        self.info_dict[LayerInfoSettings.PARENT] = self.parent
        self.info_dict[LayerInfoSettings.ISLEAF] = self.is_leaf
        self.info_dict[LayerInfoSettings.PARAMS] = self.parameters
        self.info_dict[LayerInfoSettings.BASENAME] = self.basename
        self.info_dict[LayerInfoSettings.FULLNAME] = self.full_name
        self.info_dict[LayerInfoSettings.TRAINABLE] = self.trainable
        self.info_dict[LayerInfoSettings.CLASSNAME] = self.class_name
        self.info_dict[LayerInfoSettings.INPUT_SIZE] = self.input_shape
        self.info_dict[LayerInfoSettings.NUM_PARAMS] = self.total_params
        self.info_dict[LayerInfoSettings.CHILDREN] = self.children.keys()
        self.info_dict[LayerInfoSettings.OUTPUT_SIZE] = self.output_shape
        self.info_dict[LayerInfoSettings.ORIGINALNAME] = self.original_name
        self.info_dict[LayerInfoSettings.TRAINABLE_PARAMS] = self.trainable_params
        self.info_dict[LayerInfoSettings.NON_TRAINABLE_PARAMS] = (
            self.non_trainable_params
        )
        return self.info_dict

    def __repr__(self):
        r = f"{self.original_name}---{self.full_name}---{self.basename}---{self.depth}---{self.parent}---{list(self.children.keys())}---{self.class_name}\n"
        return r
