

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