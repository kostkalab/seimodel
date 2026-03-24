import seimodel as sm
import pandas as pd
import torch
import grelu.sequence.format
import torch
import tqdm
import grelu.io.genome
import numpy as np
from importlib import resources
from pathlib import Path

try:
    from .histone_normalization import sc_hnorm_varianteffect
except ImportError:
    from histone_normalization import sc_hnorm_varianteffect


def _data_path(rel_path: str):
    try:
        return resources.files("seimodel.dat").joinpath(rel_path)
    except Exception:
        pass

    local_root = Path(__file__).resolve().parents[1] / "dat"
    candidate = local_root / rel_path
    if candidate.exists():
        return candidate

    return Path("./dist/seimodel/dat") / rel_path

# Load and cache model weights
#-----------------------------

sei_trunk_model = sm.get_sei_trunk()
sei_head_model = sm.get_sei_head()
sei_projection_model = sm.get_sei_projection()
sei_trunk_model = sei_trunk_model.load_weights()
sei_head_model = sei_head_model.load_weights()
sei_projection_model = sei_projection_model.load_weights()

# get the sequences in 1-hot
#-----------------------------

df = pd.read_csv(_data_path("vars.tsv"), sep="\t", header=None)
df.columns = ["chrom", "pos", "name", "ref", "alt"]
df.head()

genome = grelu.io.genome.get_genome("hg38")

nvars = df.shape[0]
refs = torch.empty(nvars, 4, 4096)
alts = torch.empty(nvars, 4, 4096)
labs = torch.empty(nvars)


for index, row in tqdm.tqdm(enumerate(df.itertuples(index=False)), total=nvars):
    # - variant position
    pos = row.pos - 1
    # - chromosome sequence
    chrseq = genome[row.chrom]
    assert (
        str(row.ref).upper() == str(chrseq[pos]).upper()
    ), f"Reference mismatch at chr{row.chrom}:{pos+1}"
    # - alternative sequence
    alt_seq = (
        str(chrseq[pos - 2047 : pos]) + row.alt + str(chrseq[pos + 1 : pos + 2049])
    ).upper()
    # - reference sequence
    ref_seq = str(chrseq[pos - 2047 : pos + 2049]).upper()
    #- check reference and base and base in retrieved sequence match
    assert (
        str(row.ref).upper() == ref_seq[2047]
    ), f"Reference mismatch at chr{row.chrom}:{pos+1}"
    # - one-hot encoding
    alt_one_hot = grelu.sequence.format.convert_input_type(alt_seq, "one_hot")
    alt_one_hot = alt_one_hot.unsqueeze(0)
    ref_one_hot = grelu.sequence.format.convert_input_type(ref_seq, "one_hot")
    ref_one_hot = ref_one_hot.unsqueeze(0)
    # - store
    refs[index] = ref_one_hot
    alts[index] = alt_one_hot

#- embeddings
ref_emb_fw = sei_trunk_model(refs).detach()
alt_emb_fw = sei_trunk_model(alts).detach()
ref_emb_rc = sei_trunk_model(refs.flip(dims=[1,2])).detach()
alt_emb_rc = sei_trunk_model(alts.flip(dims=[1,2])).detach()

#- chromatin features
ref_feat_fw = sei_head_model(ref_emb_fw).detach()
alt_feat_fw = sei_head_model(alt_emb_fw).detach()
ref_feat_rc = sei_head_model(ref_emb_rc).detach()
alt_feat_rc = sei_head_model(alt_emb_rc).detach()

ref_feat = (ref_feat_fw + ref_feat_rc) / 2
alt_feat = (alt_feat_fw + alt_feat_rc) / 2

#- adjustment for histone indices
histone_inds = torch.from_numpy(np.load(_data_path("histone_inds.npy")))
ref_feat_adjust, alt_feat_adjust = sc_hnorm_varianteffect(ref_feat, alt_feat, histone_inds, device="cpu")

#- sequence class scores
ref_proj = sei_projection_model(ref_feat).detach()
alt_proj = sei_projection_model(alt_feat).detach()
ref_proj_adjust = sei_projection_model(ref_feat_adjust).detach()
alt_proj_adjust = sei_projection_model(alt_feat_adjust).detach()

#- max abs difference sequence score, preserve sign
diff = alt_proj_adjust - ref_proj_adjust
scores, scores_idx = diff.abs().max(dim=1)
scores_sign = diff.sign().gather(1, scores_idx.unsqueeze(1)).squeeze(1)
final_scores = scores * scores_sign

print("Maximum absolute value difference scores computed:", final_scores)

#- load webserver sequence-class scores
wsr_path = _data_path("sei_webserver_results/a2ec9c01-7226-4c6c-b837-28dae2bfb81c_textarea_sequence-class-scores.tsv")
wsr = pd.read_csv(wsr_path, sep="\t")

#- sort the rows by position
wsr_sorted = wsr.sort_values(by="pos").reset_index(drop=True)

#- last 40 columns as numpy array
wsr_last40 = wsr_sorted.iloc[:, -40:].to_numpy()

residual = wsr_last40 - diff[:,:40].numpy() #- fourty classes

#- report maximum residual
max_residual = np.abs(residual).max()
print(f"Maximum residual between computed scores and webserver scores, for all sequence classes: {max_residual}")