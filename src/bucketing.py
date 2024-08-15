import torch
from functools import partial

# Populate with powers of 2 128 and up
bucket_sizes = []
for s in range(7,31):
    bucket_sizes.append(2**s)
def get_next_bucket_size(s):
    for b in bucket_sizes:
        if s <= b:
            return b
    raise ValueError("max bucket size exceeded")

def compactify_and_rebucket_arrays(mask, bucket_size, *arrs):
    N_in = mask.sum()
    out_mask = torch.arange(0, bucket_size) < N_in
    INVALID_IND = bucket_size + 1
    # target_inds = torch.nonzero(mask, size=bucket_size, fill_value=INVALID_IND)
    nonzero_indices = torch.nonzero(mask, as_tuple=False).flatten()
    if nonzero_indices.size(0) < bucket_size:
        num_padding = bucket_size - nonzero_indices.size(0)
        padding = torch.full((num_padding,), INVALID_IND, dtype=nonzero_indices.dtype)
        target_inds = torch.cat((nonzero_indices, padding))
    else:
        target_inds = nonzero_indices[:bucket_size]

    out_arrs = []
    for a in arrs:
        if a is None:
            out_arrs.append(a)
            continue

        if len(a.shape) == 1:
            out = torch.full((target_inds.shape[0], ), fill_value=torch.nan)
        else:
            out = torch.full((target_inds.shape[0], *a.shape[1:]), fill_value=torch.nan)
        mask = (- 1 < target_inds) & (target_inds < a.shape[0])
        inds = target_inds[mask]
        # target_inds = torch.clip(target_inds, 0, bucket_size - 1)

        # print(a.shape, out.shape)
        out[mask] = a[inds].to(dtype=out.dtype)
        out = out.squeeze(0)
        out_arrs.append(out)

    return out_mask, N_in, *out_arrs
     

def fits_in_smaller_bucket(size, curr_bucket_size):
    return get_next_bucket_size(size) < curr_bucket_size
