
## Introduction

This package downloads the Sei model weights and repackages them for easier access.  It splits the model into three sub-models (see below), making it simpler to use and providing easier access.

This repository contains all the development code needed for downloading and processing Sei and its annotations.

It also includes a script (`publish.sh`) to package the reformatted Sei model into a distribution package called `seimodel` (located in the `/dist` directory). This `seimodel` package is the intended interface to the  the repackaged Sei model.

The intended usage of `seimodel` is:

```python
import seimodel as sm

# Model classes:
sei_trunk_model = sm.get_trunk()
sei_head_model = sm.get_head()
sei_projection_model = sm.get_projection()

# Load model weights:
sei_trunk_model = sei_trunk_model.load_weights()
...
```

## Usage

1. Download the Sei model data using `./get_sei.sh`.

2. Run `./src/make_assets.py` to decompose Sei into three parts: Trunk, Head, and Projection. These are defined as follows:
    - `trunk_model`: Processes input DNA up to the smoothing and reshaping layers. Input shape is (4, 4096); output shape is (15360,).
    - `head_model`: Takes the output of Trunk and projects it to 21,907 chromatin profiles.
    - `projection_model`: Takes the output of Head and projects it onto 61 Sei sequence classes.

    The full model can be constructed as a `torch.nn.Sequential` using an `OrderedDict` of the submodels:
    ```python
    sei = torch.nn.Sequential(collections.OrderedDict([
        ('trunk', trunk_model),
        ('head', head_model),
        ('projection', projection_model)
    ]))
    ```

3. Upload the contents of the `./assets/` directory to Zenodo, where the weights are stored. The Zenodo URL is: https://zenodo.org/records/16950342

4. Run the `publish.sh` script from the project root directory to use the assets and create a clean distribution package that provides access to the three Sei models (trunk, head, projection). The distribution package components are under `./dist/seimodel`, except for the `README_dist.md` file in the root directory. See that file for more information on using the Sei models with this interface.
 
## Licensing

This repository contains code and assets under multiple licenses:

- **Original Sei model code and weights** (see `downloads/model/`, `assets/` directory after executing `get_sei.sh`) are covered by the license in `LICENSE.txt`.  In addition, the files `dist/seimodel/src/sei_trunk.py`, `dist/seimodel/src/sei_head.py`, `dist/seimodel/src/sei_projection.py` are covered by this license. Other code is not.
- **New code developed in this repository** (including scripts in `src/`, packaging scripts, and other original contributions, e.g. under `/dist/seimodel/tml_mixin`) are covered by the permissive MIT license (`LICENSE-MIT.txt`).

Please refer to each license file for details. If you use or redistribute this package, ensure you comply with the terms of both licenses.
