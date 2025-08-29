import torch
from downloads.model import sei
import gc
import numpy as np

from pathlib import Path
import hashlib

from downloads.model import sei_trunk, sei_head, sei_projection

def get_sei_model(sdfile:str, verbose:bool=True)->torch.nn.Module:
    """
    Load the SEI model from a specified state dictionary file.
    
    Args:
        sdfile (str): Path to the state dictionary file.
        
    Returns:
        torch.nn.Module: The SEI model loaded with the state dictionary.
    """
    #- sei model
    mod = sei.Sei()
    #- state dict with fitted parameters:
    sd = torch.load(sdfile, map_location='cpu')
    #- current state dict
    this_sd = mod.state_dict()
    for ky in list(this_sd.keys()):
        ky2 = 'module.model.' + ky if 'module.model.' not in ky else ky
        if ky2 in sd:
            this_sd[ky] = sd[ky2]
            if verbose:
                print(f'Applying weights for {ky} from state dict')
    mod.load_state_dict(this_sd)
    del sd, this_sd
    gc.collect()
    return mod
    
sm = get_sei_model("./model/sei.pth")
sm_sd = sm.state_dict()

#- make the sei_trunk model
#---------------------------
stm = sei_trunk.SeiTrunk()
st_sd = stm.state_dict()
for name, param in st_sd.items():
    if name in sm_sd:
        param.data.copy_(sm_sd[name].data)
        print(f'Applying weights for {name} from state dict')

stm.load_state_dict(st_sd)
stm.eval()

#- save the sei_trunk model
torch.save(stm.state_dict(), "./assets/sei_trunk.pth")

#- make the sei_head model
#---------------------------
shm = sei_head.SeiHead()
sh_sd = shm.state_dict()
for name, param in sh_sd.items():
    if name in sm_sd:
        param.data.copy_(sm_sd[name].data)
        print(f'Applying weights for {name} from state dict')

shm.load_state_dict(sh_sd) 
shm.eval()

#- save the sei_head model
torch.save(shm.state_dict(), "./assets/sei_head.pth")


#- make the projection model
#---------------------------

cent_mat = np.load('./model/projvec_targets.npy')
cent_mat_torch = torch.from_numpy(cent_mat).float()
in_features = cent_mat_torch.shape[1]
out_features = cent_mat_torch.shape[0]

spm = sei_projection.SeiProjection(n_genomic_features=in_features, n_classes=out_features)
spm.projector.weight.data = cent_mat_torch
#- save the sei_projection model
torch.save(spm.state_dict(), "./assets/sei_projection.pth")

#- calculate checksums for all the files and save into filename.sha256
#---------------------------------------------------------------------

def calculate_file_sha256(filepath: Path, chunk_size: int = 4096) -> str:
    """Calculates the SHA256 hash of a file."""
    #- assert filepath is a Path
    assert isinstance(filepath, Path), "Expected filepath to be a Path object"
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

sha = calculate_file_sha256(Path("./model/sei_projection.pth"))
with open("./assets/sei_projection.pth.sha256", "w") as f:
    f.write(sha) 
    
sha = calculate_file_sha256(Path("./model/sei_head.pth"))
with open("./assets/sei_head.pth.sha256", "w") as f:
    f.write(sha)
    
sha = calculate_file_sha256(Path("./model/sei_trunk.pth"))
with open("./assets/sei_trunk.pth.sha256", "w") as f:
    f.write(sha)

