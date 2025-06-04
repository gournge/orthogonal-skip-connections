from .resnet_base import ResNet

_VARIANTS = {
    "baseline": dict(skip_kind="identity"),
    "fully_orth": dict(skip_kind="fixed_orth"),
    "learnable_orth": dict(skip_kind="learnable_orth"),
    "partial_orth": dict(skip_kind="fixed_orth"),  # will be overridden layer‑wise
    "random": dict(skip_kind="random"),
}

def get_model(variant: str, depth: int, num_classes: int):
    if variant == "partial_orth":
        # Example: first two stages orthogonal, last random – handled via custom subclass
        from .resnet_partial import PartialOrthResNet
        return PartialOrthResNet(depth, num_classes)
    cfg = _VARIANTS[variant]
    return ResNet(depth=depth, num_classes=num_classes, **cfg)