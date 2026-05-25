import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    return sum(p.numel() for p in model.parameters())


def estimate_int8_weight_size_bytes(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Rough INT8 weight memory estimate.

    1 parameter ~= 1 byte after INT8 weight quantization.
    This does not include activation memory, code, buffers, or metadata.
    """
    return count_parameters(model, trainable_only=trainable_only)
