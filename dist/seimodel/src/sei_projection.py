"""
Sei architecture: Projection
"""
import torch.nn as nn
import torch
import numpy as np

from importlib import resources

CLASS_ANNOT_FILE = resources.files(__package__.replace('.src', '.dat')).joinpath("seqclass.names")
HISTONE_INDEX_FILE = resources.files(__package__.replace('.src', '.dat')).joinpath("histone_inds.npy")

def read_class_annot(filepath: resources.abc.Traversable) -> list[str]:
    with open(filepath, "r") as f:
        return [line.strip() for line in f] 
    

def sc_hnorm_varianteffect(chromatin_profile_ref, chromatin_profile_alt, histone_inds, device):
    histone_inds = histone_inds.clone().to(device)
    chromatin_profile_ref_adjust = chromatin_profile_ref.clone()
    chromatin_profile_ref_adjust[:, histone_inds] = chromatin_profile_ref_adjust[:, histone_inds] * (
        (chromatin_profile_ref[:, histone_inds].sum(axis=1)*0.5 +
        chromatin_profile_alt[:, histone_inds].sum(axis=1)*0.5) /
        chromatin_profile_ref[:, histone_inds].sum(axis=1))[:, None]

    chromatin_profile_alt_adjust = chromatin_profile_alt.clone()
    chromatin_profile_alt_adjust[:, histone_inds] = chromatin_profile_alt_adjust[:, histone_inds] * (
        (chromatin_profile_ref[:, histone_inds].sum(axis=1)*0.5 +
        chromatin_profile_alt[:, histone_inds].sum(axis=1)*0.5) /
        chromatin_profile_alt[:, histone_inds].sum(axis=1))[:, None]
    return (chromatin_profile_ref_adjust, chromatin_profile_alt_adjust)
    

class SeiProjection(nn.Module):

    class_annot = read_class_annot(CLASS_ANNOT_FILE)
    histone_indices = torch.from_numpy(np.load(HISTONE_INDEX_FILE))

    def __init__(self, n_genomic_features=21907, n_classes = 61):
        """
        Parameters
        ----------
        n_genomic_features : int
        n_classes : int
        """
        super(SeiProjection, self).__init__()

        self.projector = nn.Linear(n_genomic_features, n_classes, bias=False)
        self.set_mode("sequence")

    def forward(self, x):
        """Forward propagation of a batch.
        """
        if self.mode == "sequence":
            out = self.projector(x)
            return out
        elif self.mode == "variant":
            ref, alt = x
            ref_adj, alt_adj = sc_hnorm_varianteffect(ref, alt, SeiProjection.histone_indices, ref.device)
            ref_out = self.projector(ref_adj)
            alt_out = self.projector(alt_adj)
            return (ref_out, alt_out)
        else:
            print(f"Not sequence or variant, instead: {self.mode}")



    def set_mode(self, mode):
        if mode == "sequence":
            self.mode = mode
        elif mode == "variant":
            self.mode = mode
        else:
            print(f"Mode options are: \'sequence\' or \'variant\'. Keeping current mode as {self.mode}")