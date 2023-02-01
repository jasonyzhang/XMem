"""
Script with support for running XMem on multiple exemplar images.

Running script will automatically associate the corresponding images for the mask using
the file name.

Usage:
    python run_multiple_input.py --image_dir /path/to/image_dir \
        --output_dir /path/to/output_dir \
        --exemplar_masks /path/to/exemplar_mask_1.png /path/to/exemplar_mask_2.png ...
"""
import argparse
import os
import os.path as osp
from glob import glob
from typing import List, Optional

import ipdb
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from model.network import XMem
from inference.data.mask_mapper import MaskMapper
from inference.data.exemplar_dataset import ExemplarDataset
from inference.inference_core import InferenceCore


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--exemplar_masks", type=str, nargs="+", required=True)
    parser.add_argument("--size", type=int, default=480)
    return parser


def get_default_config():
    config = {
        "model": "./saves/XMem.pth",
        "disable_long_term": False,
        "enable_long_term_count_usage": True,
        "max_mid_term_frames": 10,
        "min_mid_term_frames": 5,
        "max_long_term_elements": 10000,
        "num_prototypes": 128,
        "top_k": 30,
        "mem_every": 5,
        "deep_update_every": -1,
        "flip": False,
        "size": 480,
    }
    config["enable_long_term"] = not config["disable_long_term"]
    return config


def run_xmem_exemplar(
    images: List[str],
    images_exemplar: List[str],
    masks_exemplar: List[str],
    output_dir: Optional[str] = None,
    config: Optional[dict] = None,
    network: Optional[XMem] = None,
    pbar: Optional[bool] = False,
):
    """
    Runs XMem Masks propagation given exemplar images and masks.

    If output_dir is provided, the output masks will be saved to that directory.
    Otherwise, the masks will be returned as a list.

    Args:
        images (List[str]): List of paths to images.
        images_exemplar (List[str]): List of paths to exemplar images.
        masks_exemplar (List[str]): List of paths to exemplar masks.
        output_dir (Optional[str], optional): Output directory. If None, returns a list.
        config (Optional[dict]): XMem configuration. Uses a default configuration if not
            provided. Defaults to None.
        output_dir (Optional[str], optional): Output directory. If None, returns a list
            of masks. Defaults to None.
        pbar (Optional[bool], optional): Whether to show a progress bar. Defaults to
            False.

    Returns:
        List[np.ndarray]: List of masks.
    """
    assert len(images_exemplar) == len(masks_exemplar)
    if config is None:
        config = get_default_config()
    torch.autograd.set_grad_enabled(False)

    if network is None:
        network = XMem(config, config["model"]).cuda().eval()
        model_weights = torch.load(config["model"])
        network.load_weights(model_weights, init_as_zero_if_needed=True)

    dataset = ExemplarDataset(
        images=images,
        images_exemplar=images_exemplar,
        masks_exemplar=masks_exemplar,
        size=config["size"],
    )

    config["num_input"] = len(images_exemplar)
    mapper = MaskMapper()
    processor = InferenceCore(network, config=config)
    num_mask_seen = 0

    vid_length = len(dataset)
    pred_masks = []

    loop = tqdm(enumerate(dataset), total=vid_length) if pbar else enumerate(dataset)
    for index, data in loop:
        with torch.cuda.amp.autocast(enabled=True):
            rgb = data["rgb"].cuda()
            msk = data.get("mask")
            info = data["info"]
            shape = info["shape"]
            need_resize = info["need_resize"]

            if msk is not None:
                num_mask_seen += 1

            if num_mask_seen == 0:
                # No point doing anything without a mask
                continue

            if config["flip"]:
                rgb = torch.flip(rgb, dims=[-1])
                msk = torch.flip(msk, dims=[-1]) if msk is not None else None

            # Map possibly non-continuous labels to continuous ones
            if msk is not None:
                msk, labels = mapper.convert_mask(msk, exhaustive=True)
                msk = torch.Tensor(msk).cuda()
                if msk.sum() == 0:
                    # Mask is empty!
                    print("Skipping empty mask, decrementing num input")
                    processor.increment_num_input(-1)
                    num_mask_seen -= 1
                    continue
                if need_resize:
                    msk = dataset.resize_mask(msk.unsqueeze(0))[0]
                processor.set_all_labels(list(mapper.remappings.values()))
            else:
                labels = None

            prob = processor.step(rgb, msk, labels, end=(index == vid_length - 1))

            if need_resize:
                prob = F.interpolate(
                    prob.unsqueeze(1), shape, mode="bilinear", align_corners=False
                )[:, 0]

            if config["flip"]:
                prob = torch.flip(prob, dims=[-1])

            # Probability mask -> index mask
            out_mask = torch.argmax(prob, dim=0)
            out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

            # Save the mask
            if output_dir is not None:
                new_name = osp.join(output_dir, info["name"] + ".png")
                os.makedirs(output_dir, exist_ok=True)
                out_mask = mapper.remap_index_mask(out_mask)
                out_img = Image.fromarray(out_mask)
                if dataset.get_palette() is not None:
                    out_img.putpalette(dataset.get_palette())
                out_img.save(new_name)
            else:
                pred_masks.append(out_mask)
    if output_dir is None:
        return pred_masks


if __name__ == "__main__":
    args = get_parser().parse_args()
    config = get_default_config()
    config["size"] = args.size

    exemplar_masks = args.exemplar_masks
    all_images = sorted(glob(osp.join(args.image_dir, "*.jpg")))
    images_map = {osp.splitext(osp.basename(x))[0]: x for x in all_images}
    exemplar_images = [
        images_map[osp.splitext(osp.basename(x))[0]] for x in exemplar_masks
    ]
    run_xmem_exemplar(
        images=all_images,
        images_exemplar=exemplar_images,
        masks_exemplar=exemplar_masks,
        output_dir=args.output_dir,
        config=config,
    )
