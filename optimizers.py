# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.optim as optim


def exclude_bias_and_norm(p):
    return p.ndim == 1


class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def build_optimizer(args, model):
    if args.optimizer == "adamw":

        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            maps_group = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if "maps_classifier" in name:
                    maps_group.append(param)
                    continue

                if len(param.shape) == 1 or name in skip_list:
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {"params": no_decay, "weight_decay": 0.0},
                {"params": decay, "weight_decay": weight_decay},
                {
                    "params": maps_group,
                    "weight_decay": weight_decay,
                    "__MAPS_TOKEN__": "",
                },
            ]

        parameters = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.AdamW(parameters, weight_decay=0.0)
    elif args.optimizer == "lars":
        optimizer = LARS(
            model.parameters(),
            lr=None,
            weight_decay=1e-6,
            momentum=0.9,
            eta=1e-3,
            weight_decay_filter=exclude_bias_and_norm,
            lars_adaptation_filter=exclude_bias_and_norm,
        )

    return optimizer
