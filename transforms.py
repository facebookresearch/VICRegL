# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
import numbers
import warnings
from collections.abc import Sequence
import numpy as np
from PIL import ImageOps, ImageFilter

import torch

import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import _interpolation_modes_from_int
import torchvision.transforms.functional as FT


class GaussianBlur(object):
    def __init__(self):
        pass

    def __call__(self, img):
        sigma = np.random.rand() * 1.9 + 0.1
        return img.filter(ImageFilter.GaussianBlur(sigma))


class Solarization(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return ImageOps.solarize(img)


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class RandomResizedCropWithLocation(torch.nn.Module):
    """Crop the given image to random size and aspect ratio.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size (int or sequence): expected output size of each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
            In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        scale (tuple of float): scale range of the cropped image before resizing, relatively to the origin image.
        ratio (tuple of float): aspect ratio range of the cropped image before resizing.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
            ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.

    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=InterpolationMode.BILINEAR,
    ):
        super().__init__()
        self.size = _setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        )

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = FT.get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w, height, width

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w, height, width

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w, H, W = self.get_params(img, self.scale, self.ratio)
        return (
            FT.resized_crop(img, i, j, h, w, self.size, self.interpolation),
            (i, j, h, w, H, W),
        )

    def __repr__(self):
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + "(size={0}".format(self.size)
        format_string += ", scale={0}".format(tuple(round(s, 4) for s in self.scale))
        format_string += ", ratio={0}".format(tuple(round(r, 4) for r in self.ratio))
        format_string += ", interpolation={0})".format(interpolate_str)
        return format_string


class RandomHorizontalFlipReturnsIfFlip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return FT.hflip(img), True
        return img, False

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


def _location_to_NxN_grid(location, N=7, flip=False):
    i, j, h, w, H, W = location
    size_h_case = h / N
    size_w_case = w / N
    half_size_h_case = size_h_case / 2
    half_size_w_case = size_w_case / 2
    final_grid_x = torch.zeros(N, N)
    final_grid_y = torch.zeros(N, N)

    final_grid_x[0][0] = i + half_size_h_case
    final_grid_y[0][0] = j + half_size_w_case
    for k in range(1, N):
        final_grid_x[k][0] = final_grid_x[k - 1][0] + size_h_case
        final_grid_y[k][0] = final_grid_y[k - 1][0]
    for l in range(1, N):
        final_grid_x[0][l] = final_grid_x[0][l - 1]
        final_grid_y[0][l] = final_grid_y[0][l - 1] + size_w_case
    for k in range(1, N):
        for l in range(1, N):
            final_grid_x[k][l] = final_grid_x[k - 1][l] + size_h_case
            final_grid_y[k][l] = final_grid_y[k][l - 1] + size_w_case

    final_grid = torch.stack([final_grid_x, final_grid_y], dim=-1)
    if flip:
        # start_grid = final_grid.clone()
        for k in range(0, N):
            for l in range(0, N // 2):
                swap = final_grid[k, l].clone()
                final_grid[k, l] = final_grid[k, N - 1 - l]
                final_grid[k, N - 1 - l] = swap

    return final_grid


def get_color_distortion(left=True):
    if left:
        p_blur = 1.0
        p_sol = 0.0
    else:
        p_blur = 0.1
        p_sol = 0.2
    # s is the strength of color distortion.
    transform = transforms.Compose(
        [
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=p_blur),
            transforms.RandomApply([Solarization()], p=p_sol),
        ]
    )
    return transform


class MultiCropTrainDataTransform(object):
    def __init__(
        self,
        size_crops=[224, 96],
        num_crops=[2, 6],
        min_scale_crops=[0.14, 0.05],
        max_scale_crops=[1, 0.14],
        return_location_masks=False,
        no_flip_grid=False,
    ):
        self.size_crops = size_crops
        self.num_crops = num_crops
        self.return_location_masks = return_location_masks
        self.no_flip_grid = no_flip_grid

        self.random_resized_crops = []
        self.augmentations = []
        self.flip = RandomHorizontalFlipReturnsIfFlip(p=0.5)
        for i in range(len(size_crops)):
            for j in range(num_crops[i]):
                if self.return_location_masks:
                    random_resized_crop = RandomResizedCropWithLocation(
                        size_crops[i],
                        scale=(min_scale_crops[i], max_scale_crops[i]),
                        interpolation=InterpolationMode.BICUBIC,
                    )
                else:
                    random_resized_crop = transforms.RandomResizedCrop(
                        size_crops[i],
                        scale=(min_scale_crops[i], max_scale_crops[i]),
                        interpolation=InterpolationMode.BICUBIC,
                    )
                self.random_resized_crops.append(random_resized_crop)
                self.augmentations.append(
                    transforms.Compose(
                        [
                            get_color_distortion(left=(j % 2 == 0)),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225],
                            ),
                        ]
                    )
                )

    def __call__(self, img):
        multi_crops_no_augs = list(
            map(lambda trans: trans(img), self.random_resized_crops)
        )
        if self.return_location_masks:
            multi_crops = []
            locations = []
            for i, (crop, location) in enumerate(multi_crops_no_augs):
                crop, is_flip = self.flip(crop)
                multi_crops.append(self.augmentations[i](crop))
                grid_size = 7
                if i >= self.num_crops[0]:
                    grid_size = 3
                if self.no_flip_grid:
                    is_flip = False
                locations.append(
                    _location_to_NxN_grid(location, grid_size, flip=is_flip)
                )

            return multi_crops, locations
        multi_crops = [
            self.augmentations[i](crop) for i, crop in enumerate(multi_crops_no_augs)
        ]
        return multi_crops


class MultiCropValDataTransform(MultiCropTrainDataTransform):
    def __init__(self, **kw):
        super().__init__(**kw)
        if self.size_crops[0] == 224:
            full_size = 256
        elif self.size_crops[0] == 384:
            full_size = 438
        self.eval_trans = transforms.Compose(
            [
                transforms.Resize(full_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.size_crops[0]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, img):
        val_crop = self.eval_trans(img)
        val_crop_with_train_transform = super().__call__(img)
        return (val_crop, val_crop_with_train_transform)
