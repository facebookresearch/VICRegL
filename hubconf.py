# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import resnet
import convnext

dependencies = ["torch", "torchvision"]


def resnet50_alpha0p9(pretrained=True, **kwargs):
    model, _ = resnet.resnet50(**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/vicregl/resnet50_alpha0.9.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def resnet50_alpha0p75(pretrained=True, **kwargs):
    model, _ = resnet.resnet50(**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/vicregl/resnet50_alpha0.75.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def convnext_small_alpha0p9(pretrained=True, **kwargs):
    model, _ = convnext.convnext_small(**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/vicregl/convnext_small_alpha0.9.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def convnext_small_alpha0p75(pretrained=True, **kwargs):
    model, _ = convnext.convnext_small(**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/vicregl/convnext_small_alpha0.75.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def convnext_base_alpha0p9(pretrained=True, **kwargs):
    model, _ = convnext.convnext_base(**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/vicregl/convnext_base_alpha0.9.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def convnext_base_alpha0p75(pretrained=True, **kwargs):
    model, _ = convnext.convnext_base(**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/vicregl/convnext_base_alpha0.75.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def convnext_xlarge_alpha0p75(pretrained=True, **kwargs):
    model, _ = convnext.convnext_xlarge(**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/vicregl/convnext_xlarge_alpha0.75.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model
