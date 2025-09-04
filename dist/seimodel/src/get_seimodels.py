from .tml_mixin.mixin import TorchModelLoaderMixin
from .sei_trunk import SeiTrunk
from .sei_head import SeiHead
from .sei_projection import SeiProjection
import yaml
import torch
from importlib import resources

CONFIG_FILE = resources.files("seimodel.dat").joinpath("config.yaml")


# - read in configuration
with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

BASE_URL = config["base_url"]
APP_NAME = config["app_name"]
VERSION = str(config["version"])


def make_loadable(mixin, base):
    class Loadable(mixin, base):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    return Loadable


SeiTrunkLoadable = make_loadable(TorchModelLoaderMixin, SeiTrunk)
SeiHeadLoadable = make_loadable(TorchModelLoaderMixin, SeiHead)
SeiProjectionLoadable = make_loadable(TorchModelLoaderMixin, SeiProjection)

MODEL_PARAMS = {
    "trunk": {'class': SeiTrunkLoadable,
              'filename': config["filename_trunk"]},
    "head": {'class': SeiHeadLoadable,
             'filename': config["filename_head"]},
    "projection": {'class': SeiProjectionLoadable,
                   'filename': config["filename_projection"]},
}


def get_sei_model(model_name: str, load_weights: bool = False) -> torch.nn.Module:
    model_cls = MODEL_PARAMS[model_name]['class']
    model = model_cls(
        base_url=BASE_URL, filename=MODEL_PARAMS[model_name]['filename'], app_name=APP_NAME, version=VERSION
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
