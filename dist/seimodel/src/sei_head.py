"""
Sei architecture: Head
"""

import torch.nn as nn
from scipy.interpolate import splev

from importlib import resources
from importlib.abc import Traversable
import re

# - get the chromatin profile annotations
TARGET_ANNOT_FILE = resources.files(__package__.replace('.src', '.dat')).joinpath("target.names")

def read_target_annot(filepath: Traversable) -> dict[str, list[str]]:
    target_annot = {"context": [], "assay": [], "info": []}
    try:
        with open(filepath, "r") as f:
            for i, line in enumerate(f):
                parts = line.strip().split(" | ")
                if len(parts) in [3, 4]:  # - some lines have 4 fields, probably to make it uniq.
                    target_annot["context"].append(parts[0])
                    target_annot["assay"].append(parts[1])
                    target_annot["info"].append(parts[2])
                else:
                    raise ValueError(f"Malformed line in '{TARGET_ANNOT_FILE}' at line {i + 1}")
    except FileNotFoundError:
        print(f"Error: The file '{TARGET_ANNOT_FILE}' was not found.")
    return target_annot


class SeiHead(nn.Module):

    # - add target annotation as class attribute
    target_annot = read_target_annot(TARGET_ANNOT_FILE)

    def __init__(self, dim_ipt=15360, n_genomic_features=21907):
        super(SeiHead, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(dim_ipt, n_genomic_features),
            nn.ReLU(inplace=True),
            nn.Linear(n_genomic_features, n_genomic_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        predict = self.classifier(x)
        return predict

    def search_target_annot(
        self, pattern: str, field: str = "context", return_annot: bool = False
    ) -> list[int] | tuple[list[int], dict[str, list[str]]]:
        """
        Search for a regex pattern in the specified field of target_annot.
        Returns a list of indices where the pattern matches.
        If return_annot is True, also returns a dict with annotations for
        the matching entries.
        """

        regex = re.compile(pattern)
        # Assume target_annot is a list of dicts with keys: 'context', 'assay', 'info'
        matches = []
        for idx, annot in enumerate(self.target_annot):
            # - fixme: seems horribly inefficient to seperately search each field...
            if field in annot and regex.search(str(annot[field])):
                matches.append(idx)
        if return_annot:
            # Slice all lists in the dict to only matching indices
            matched = {k: [v[i] for i in matches] for k, v in self.target_annot.items()}
            return matches, matched
        return matches
