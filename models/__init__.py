from .resnet_orthogonal import ResNetOrthogonal
from .resnet_learnable_orthogonal import ResNetLearnableOrthogonal
from .resnet_partial_orthogonal import ResNetPartialOrthogonal
from .resnet_random_skip import ResNetRandomSkip
from .resnet_baseline import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

_MODEL_REGISTRY = {
    "orthogonal": ResNetOrthogonal,
    "learnable_ortho": ResNetLearnableOrthogonal,
    "partial_ortho": ResNetPartialOrthogonal,
    "random_skip": ResNetRandomSkip,
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
}


def get_model(name: str, **kwargs):
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'")
    return _MODEL_REGISTRY[name](**kwargs)
