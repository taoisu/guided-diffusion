"""
Train a super-resolution model.
"""

import argparse
from copy import deepcopy

import torch as th
import torch.nn.functional as F

from einops import rearrange

from PIL import Image

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def sample_fn(model, diffusion, batch, cond):
    cond = { k: v.to(dist_util.dev()) for k, v in cond.items() }
    diffusion = deepcopy(diffusion)
    samples = diffusion.ddim_sample_loop(
        model=model,
        shape=batch.shape,
        model_kwargs=cond,
        progress=True,
    )
    samples = ((samples+1)*127.5).clamp(0,255).to(th.uint8)
    samples = rearrange(samples, 'b c h w -> b h w c')
    b, h, w, _ = samples.shape
    step = 4
    ws = w * step
    hs = (b + step - 1) // step * h
    final_image = Image.new('RGB', (ws, hs), 'white')
    for i, sample in enumerate(samples):
        x, y = (i % step) * w, (i // step) * h
        image = Image.fromarray(sample.cpu().numpy())
        final_image.paste(image, (x, y))
    return final_image


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_superres_data(
        args.data_dir,
        args.batch_size,
        large_size=args.large_size,
        small_size=args.small_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop(sample_fn)


def load_superres_data(data_dir, batch_size, large_size, small_size, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=class_cond,
        random_crop=True,
    )
    for large_batch, model_kwargs in data:
        model_kwargs["low_res"] = F.interpolate(large_batch, small_size, mode="area")
        yield large_batch, model_kwargs


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=2000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        use_new_attention_order=False,
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
