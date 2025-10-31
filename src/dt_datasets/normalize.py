# This file supplies action normalization utilities for DeepThinkVLA.
# Author: Cheng Yin
# Date: 2025-09
# Copyright (c) Cheng Yin. All rights reserved.
# See LICENSE file in the project root for license information.

import numpy as np
import torch
from torch import Tensor, nn


def create_stats_buffers(
    normalization_type: str,
    stats = None,
):
    assert isinstance(normalization_type, str)

    shape = tuple(stats["mean"].shape)
    if normalization_type == "MEAN_STD":
        mean = torch.ones(shape, dtype=torch.float32) * torch.inf
        std = torch.ones(shape, dtype=torch.float32) * torch.inf
        buffer = nn.ParameterDict(
            {
                "mean": nn.Parameter(mean, requires_grad=False),
                "std": nn.Parameter(std, requires_grad=False),
            }
        )
    elif normalization_type == "MIN_MAX":
        min = torch.ones(shape, dtype=torch.float32) * torch.inf
        max = torch.ones(shape, dtype=torch.float32) * torch.inf
        buffer = nn.ParameterDict(
            {
                "min": nn.Parameter(min, requires_grad=False),
                "max": nn.Parameter(max, requires_grad=False),
            }
        )
    elif normalization_type == "QUANTILE":
        q01 = torch.ones(shape, dtype=torch.float32) * torch.inf
        q99 = torch.ones(shape, dtype=torch.float32) * torch.inf
        buffer = nn.ParameterDict(
            {
                "q01": nn.Parameter(q01, requires_grad=False),
                "q99": nn.Parameter(q99, requires_grad=False),
            }
        )

    if stats:
        if isinstance(stats["mean"], np.ndarray):
            if normalization_type == "MEAN_STD":
                buffer["mean"].data = torch.from_numpy(stats["mean"]).to(dtype=torch.float32)
                buffer["std"].data = torch.from_numpy(stats["std"]).to(dtype=torch.float32)
            elif normalization_type == "MIN_MAX":
                buffer["min"].data = torch.from_numpy(stats["min"]).to(dtype=torch.float32)
                buffer["max"].data = torch.from_numpy(stats["max"]).to(dtype=torch.float32)
            elif normalization_type == "QUANTILE":
                buffer["q01"].data = torch.from_numpy(stats["q01"]).to(dtype=torch.float32)
                buffer["q99"].data = torch.from_numpy(stats["q99"]).to(dtype=torch.float32)

        elif isinstance(stats["mean"], torch.Tensor):
            if normalization_type == "MEAN_STD":
                buffer["mean"].data = stats["mean"].clone().to(dtype=torch.float32)
                buffer["std"].data = stats["std"].clone().to(dtype=torch.float32)
            elif normalization_type == "MIN_MAX":
                buffer["min"].data = stats["min"].clone().to(dtype=torch.float32)
                buffer["max"].data = stats["max"].clone().to(dtype=torch.float32)
            elif normalization_type == "QUANTILE":
                buffer["q01"].data = stats["q01"].clone().to(dtype=torch.float32)
                buffer["q99"].data = stats["q99"].clone().to(dtype=torch.float32)
        else:
            type_ = type(stats["mean"])
            raise ValueError(f"np.ndarray or torch.Tensor expected, but type is '{type_}' instead.")

    return buffer


def _no_stats_error_str(name: str) -> str:
    return (
        f"`{name}` is infinity. You should either initialize with `stats` as an argument, or use a "
        "pretrained model."
    )


class Normalize_Action(nn.Module):
    """Normalizes data (e.g. "observation.image") for more stable and faster convergence during training."""

    def __init__(
        self,
        normalization_type: str,
        stats = None,
        action_mask = None,
    ):
        super().__init__()
        self.normalization_type = normalization_type
        self.stats = stats
        self.action_mask = torch.tensor(action_mask, dtype=torch.bool) if action_mask is not None else None
        stats_buffers = create_stats_buffers(normalization_type, stats)
        setattr(self, "buffer_value", stats_buffers)

    # TODO(rcadene): should we remove torch.no_grad?
    @torch.no_grad
    def forward(self, action):
        self.action_mask = self.action_mask.to(action.device)
        buffer = getattr(self, "buffer_value")

        if self.normalization_type == "MEAN_STD":
            mean = buffer["mean"].to(action.device, dtype=action.dtype)
            std = buffer["std"].to(action.device, dtype=action.dtype)
            assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
            assert not torch.isinf(std).any(), _no_stats_error_str("std")
            normalized_actions = torch.where(
                self.action_mask,
                (action - mean) / (std + 1e-8),
                action,
            )
        elif self.normalization_type == "MIN_MAX":
            min = buffer["min"].to(action.device, dtype=action.dtype)
            max = buffer["max"].to(action.device, dtype=action.dtype)
            assert not torch.isinf(min).any(), _no_stats_error_str("min")
            assert not torch.isinf(max).any(), _no_stats_error_str("max")
            normalized_actions = torch.where(
                self.action_mask,
                torch.clamp(2 * (action - min) / (max - min + 1e-8) - 1, -1, 1),
                action,
            )
        elif self.normalization_type == "QUANTILE":
            q01 = buffer["q01"].to(action.device, dtype=action.dtype)
            q99 = buffer["q99"].to(action.device, dtype=action.dtype)
            assert not torch.isinf(q01).any(), _no_stats_error_str("q01")
            assert not torch.isinf(q99).any(), _no_stats_error_str("q99")
            normalized_actions = torch.where(
                self.action_mask,
                torch.clamp(2 * (action - q01) / (q99 - q01 + 1e-8) - 1, -1, 1),
                action,
            )

        else:
            raise ValueError(self.normalization_type)
        return normalized_actions


class Unnormalize_Action(nn.Module):
    def __init__(
        self,
        normalization_type: str,
        stats = None,
        action_mask = None,
    ):
        super().__init__()
        self.normalization_type = normalization_type
        self.stats = stats
        self.action_mask = torch.tensor(action_mask, dtype=torch.bool) if action_mask is not None else None
        stats_buffers = create_stats_buffers(normalization_type, stats)
        setattr(self, "buffer_value", stats_buffers)

    # TODO(rcadene): should we remove torch.no_grad?
    @torch.no_grad
    def forward(self, normalized_actions):
        self.action_mask = self.action_mask.to(normalized_actions.device)
        buffer = getattr(self, "buffer_value")

        if self.normalization_type == "MEAN_STD":
            mean = buffer["mean"].to(normalized_actions.device, dtype=normalized_actions.dtype)
            std = buffer["std"].to(normalized_actions.device, dtype=normalized_actions.dtype)
            assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
            assert not torch.isinf(std).any(), _no_stats_error_str("std")
            action = torch.where(
                self.action_mask,
                normalized_actions * std + mean,
                normalized_actions,
            )
        elif self.normalization_type == "MIN_MAX":
            min = buffer["min"].to(normalized_actions.device, dtype=normalized_actions.dtype)
            max = buffer["max"].to(normalized_actions.device, dtype=normalized_actions.dtype)
            assert not torch.isinf(min).any(), _no_stats_error_str("min")
            assert not torch.isinf(max).any(), _no_stats_error_str("max")
            action = torch.where(
                self.action_mask,
                0.5 * (normalized_actions + 1) * (max - min + 1e-8) + min,
                normalized_actions,
            )
        elif self.normalization_type == "QUANTILE":
            q01 = buffer["q01"].to(normalized_actions.device, dtype=normalized_actions.dtype)
            q99 = buffer["q99"].to(normalized_actions.device, dtype=normalized_actions.dtype)
            assert not torch.isinf(q01).any(), _no_stats_error_str("q01")
            assert not torch.isinf(q99).any(), _no_stats_error_str("q99")
            action = torch.where(
                self.action_mask,
                0.5 * (normalized_actions + 1) * (q99 - q01 + 1e-8) + q01,
                normalized_actions,
            )
        else:
            raise ValueError(self.normalization_type)
        return action
