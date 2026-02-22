from orthogonal_skip_connections.models.resnet_base import (
    resnet18,
    resnet34,
    resnet50,
    resnet56,
)
from orthogonal_skip_connections.models.skip import (
    get_skip_names,
)

_BASE_MODELS = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet56": resnet56,
}


def get_model(
    num_classes: int,
    model_type: str,
    skip_kind: str = "identity",
    update_rule: str | None = None,
    update_rule_iters: int | None = None,
):
    if skip_kind not in get_skip_names():
        raise ValueError(
            f"Invalid skip type '{skip_kind}'. Available types: {get_skip_names()}"
        )

    if model_type not in _BASE_MODELS:
        raise ValueError(
            f"Model size '{model_type}' is not supported. Available sizes: {list(_BASE_MODELS.keys())}"
        )

    return _BASE_MODELS[model_type](
        num_classes=num_classes,
        skip_kind=skip_kind,
        update_rule=update_rule,
        update_rule_iters=update_rule_iters,
    )
