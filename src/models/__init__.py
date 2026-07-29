from .resunet import ResUNet
from .siglip2_unet import Siglip2UNet

def build_model(cfg: dict):
    m = cfg["model"]
    name = m["name"]
    if name == "resunet":
        return ResUNet(
            in_channels=int(m.get("in_channels", 3)),
            out_channels=int(m.get("out_channels", 1)),
            base_channels=int(m.get("base_channels", 32)),
        )
    if name == "siglip2_unet":
        return Siglip2UNet(
            checkpoint=m["checkpoint"],
            feature_layers=list(m.get("feature_layers", [3, 6, 9, 12])),
            decoder_channels=int(m.get("decoder_channels", 256)),
            out_channels=int(m.get("out_channels", 1)),
            train_mode=m.get("train_mode", "full"),
            partial_last_n=int(m.get("partial_last_n", 4)),
        )
    raise ValueError(f"Unknown model: {name}")
