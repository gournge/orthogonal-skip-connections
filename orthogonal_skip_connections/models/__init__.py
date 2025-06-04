from .resnet_base import ResNet
from .resnet_std import resnet18, resnet50

_VARIANTS = {
    "baseline": dict(skip_kind="identity"),
    "fully_orth": dict(skip_kind="fixed_orth"),
    "learnable_orth": dict(skip_kind="learnable_orth"),
    "partial_orth": dict(skip_kind="fixed_orth"),  # will be overridden layer-wise
    "random": dict(skip_kind="random"),
}

_STD_MODELS = {
    "resnet18": resnet18,
    "resnet50": resnet50,
}

def get_model(variant: str, depth: int, num_classes: int):
    if variant in _STD_MODELS:
        return _STD_MODELS[variant](num_classes=num_classes)
    if variant == "partial_orth":
        from .resnet_partial import PartialOrthResNet
        return PartialOrthResNet(depth, num_classes)
    cfg = _VARIANTS[variant]
    return ResNet(depth=depth, num_classes=num_classes, **cfg)