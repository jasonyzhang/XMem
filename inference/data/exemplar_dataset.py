import os.path as osp

import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from dataset.range_transform import im_normalization


class ExemplarDataset(Dataset):
    """
    Dataset for a single video. First outputs exemplar frames before going through all
    images that need masks.
    """

    def __init__(
        self,
        images,
        images_exemplar,
        masks_exemplar,
        size=-1,
    ):
        self.images = images
        self.images_exemplar = images_exemplar
        self.masks_exemplar = masks_exemplar

        self.palette = Image.open(self.masks_exemplar[0]).getpalette()
        if size < 0:
            self.im_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    im_normalization,
                ]
            )
        else:
            self.im_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    im_normalization,
                    transforms.Resize(size, interpolation=InterpolationMode.BILINEAR),
                ]
            )
        self.size = size
        self.first_shape = None

    def __len__(self):
        return len(self.images) + len(self.images_exemplar)

    def __getitem__(self, idx):
        if idx < len(self.images_exemplar):
            image_path = self.images_exemplar[idx]
            mask_path = self.masks_exemplar[idx]
        else:
            image_path = self.images[idx - len(self.images_exemplar)]
            mask_path = None

        image = Image.open(image_path).convert("RGB")
        if self.first_shape is None:
            self.first_shape = image.size
        info = {
            "frame": osp.basename(image_path),
            "name": osp.splitext(osp.basename(image_path))[0],
            "need_resize": not (self.size < 0),
            "shape": np.array(image).shape[:2],
        }
        image = image.resize(self.first_shape, Image.BILINEAR)
        image = self.im_transform(image)
        data = {
            "rgb": image,
            "info": info,
        }

        if mask_path is not None:
            mask = Image.open(mask_path).convert("P")
            mask = mask.resize(self.first_shape, Image.BILINEAR)
            mask = np.array(mask, dtype=np.uint8)
            data["mask"] = mask

        return data

    def get_palette(self):
        return self.palette

    def resize_mask(self, mask):
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return F.interpolate(
            mask,
            (int(h / min_hw * self.size), int(w / min_hw * self.size)),
            mode="nearest",
        )
