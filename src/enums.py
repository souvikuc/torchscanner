from enum import Enum, StrEnum, unique


@unique
class LayerInfoSettings(StrEnum):
    """Enum containing all available column settings."""

    __slots__ = ()

    DEPTH = "depth"
    PARENT = "parent"
    ISLEAF = "is_leaf"
    CHILDREN = "children"
    BASENAME = "basename"
    PARAMS = "parameters"
    FULLNAME = "full_name"
    TRAINABLE = "trainable"
    CLASSNAME = "class_name"
    INPUT_SIZE = "input_shape"
    NUM_PARAMS = "total_params"
    OUTPUT_SIZE = "output_shape"
    ORIGINALNAME = "original_name"
    TRAINABLE_PARAMS = "trainable_params"
    NON_TRAINABLE_PARAMS = "non_trainable_params"
    # PARAMS_PERCENT = "params_percent"
    # MULT_ADDS = "mult_adds"
