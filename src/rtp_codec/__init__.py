"""RTP-CODEC: compact sequence-only TCR representations."""
__version__ = "0.1.0"
__all__ = ["ForwardModel", "InverseModel", "RTPCodecConfig", "RTPCodecTransformer"]

def __getattr__(name):
    if name == "ForwardModel":
        from .models.forward import ForwardModel
        return ForwardModel
    if name == "InverseModel":
        from .models.inverse import InverseModel
        return InverseModel
    if name in {"RTPCodecConfig", "RTPCodecTransformer"}:
        from .models.codec import RTPCodecConfig, RTPCodecTransformer
        return {"RTPCodecConfig": RTPCodecConfig, "RTPCodecTransformer": RTPCodecTransformer}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
