from .tml_mixin.mixin import TorchModelLoaderMixin
from sei_trunk import SeiTrunk
from sei_head import SeiHead
from sei_projection import SeiProjection
import yaml
import torch
from importlib import resources

CONFIG_FILE = resources.files(__package__).joinpath("data", "config.yaml")


# - read in configuration
with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

BASE_URL = config["base_url"]
APP_NAME = config["app_name"]
FILE_NAME = config["filename_trunk"]
VERSION = config["version"]


def make_loadable(mixin, base):
    class Loadable(mixin, base):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    return Loadable


SeiTrunkLoadable = make_loadable(TorchModelLoaderMixin, SeiTrunk)
SeiHeadLoadable = make_loadable(TorchModelLoaderMixin, SeiHead)
SeiProjectionLoadable = make_loadable(TorchModelLoaderMixin, SeiProjection)

MODEL_CLASSES = {
    "trunk": SeiTrunkLoadable,
    "head": SeiHeadLoadable,
    "projection": SeiProjectionLoadable,
}


def get_sei_model(model_name: str, load_weights: bool = False) -> torch.nn.Module:
    model_cls = MODEL_CLASSES[model_name]
    model = model_cls(
        base_url=BASE_URL, filename=FILE_NAME, app_name=APP_NAME, version=VERSION
    )
    if load_weights:
        model = model.load_weights()
    return model

# For easier use:
def get_sei_trunk(load_weights: bool = False) -> torch.nn.Module:
    return get_sei_model("trunk", load_weights)


def get_sei_head(load_weights: bool = False) -> torch.nn.Module:
    return get_sei_model("head", load_weights)


def get_sei_projection(load_weights: bool = False) -> torch.nn.Module:
    return get_sei_model("projection", load_weights)
