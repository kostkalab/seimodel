"""
Sei architecture: Projection
"""
import torch.nn as nn

from importlib import resources

CLASS_ANNOT_FILE = resources.files(__package__).joinpath("dat", "seqclass.names")

def read_class_annot(filepath: resources.abc.Traversable) -> list[str]:
    with open(filepath, "r") as f:
        return [line.strip() for line in f] 

class SeiProjection(nn.Module):

    class_annot = read_class_annot(CLASS_ANNOT_FILE)

    def __init__(self, n_genomic_features=21907, n_classes = 61):
        """
        Parameters
        ----------
        n_genomic_features : int
        n_classes : int
        """
        super(SeiProjection, self).__init__()

        self.projector = nn.Linear(n_genomic_features, n_classes, bias=False)

    def forward(self, x):
        """Forward propagation of a batch.
        """
        out = self.projector(x)
        return out
