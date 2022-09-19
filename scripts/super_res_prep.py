import fire

import numpy as np
import torch.nn.functional as F

from einops import rearrange
from guided_diffusion.image_datasets import load_data


def load_superres_data(data_dir, batch_size, large_size, small_size, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=class_cond,
    )
    for large_batch, model_kwargs in data:
        model_kwargs["low_res"] = F.interpolate(large_batch, small_size, mode="area")
        yield large_batch, model_kwargs


def main(
    data_dir:str,
    out_path:str,
    num_samples:int=8,
    large_size:int=512,
    small_size:int=128,
):
    data = load_superres_data(
        data_dir=data_dir,
        batch_size=num_samples,
        large_size=large_size,
        small_size=small_size,
    )
    item = next(data)
    batch = rearrange((item[1]['low_res']+1)*127.5, 'b c h w -> b h w c')
    print(batch.shape)
    batch = batch.numpy().astype(np.uint8)
    np.savez(out_path, arr_0=batch)


if __name__ == '__main__':
    fire.Fire(main)