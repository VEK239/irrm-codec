"""IRRM-CODEC package."""

__all__ = ["ForwardModel", "InverseModel", "IRRMCodecConfig", "IRRMCodecTransformer"]


def __getattr__(name):
    if name == "ForwardModel":
        from .forward_model import ForwardModel

        return ForwardModel
    if name == "InverseModel":
        from .inverse_model import InverseModel

        return InverseModel
    if name in {"IRRMCodecConfig", "IRRMCodecTransformer"}:
        from .multitask_transformer import IRRMCodecConfig, IRRMCodecTransformer

        return {
            "IRRMCodecConfig": IRRMCodecConfig,
            "IRRMCodecTransformer": IRRMCodecTransformer,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
