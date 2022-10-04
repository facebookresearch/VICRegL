# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


_base_ = [
    "../_base_/datasets/pascal_voc12_aug.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_40k.py",
]
crop_size = (512, 512)

norm_cfg = dict(type="SyncBN", requires_grad=True)
model = dict(
    type="EncoderDecoder",
    pretrained=None,
    backbone=dict(
        type="ConvNeXt",
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
        freeze_weights=True,
    ),
    decode_head=dict(
        type="LinearHead",
        in_channels=[768, 384, 192, 96],
        in_index=[0, 1, 2, 3],
        input_transform="resize_concat",
        channels=1440,
        dropout_ratio=0,
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    test_cfg=dict(mode="slide", crop_size=crop_size, stride=(341, 341)),
    init_cfg=dict(type="Pretrained", checkpoint=""),
)

optimizer = dict(
    _delete_=True, type="AdamW", lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05
)

lr_config = dict(
    _delete_=True,
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
