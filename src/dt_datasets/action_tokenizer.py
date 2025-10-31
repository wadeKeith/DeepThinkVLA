# This file implements the action tokenizer wrapper for DeepThinkVLA.
# Author: Cheng Yin
# Date: 2025-09
# Copyright (c) Cheng Yin. All rights reserved.
# See LICENSE file in the project root for license information.

"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
"""

from typing import List, Union

import torch
from transformers import PreTrainedTokenizerBase


class ActionTokenizer:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, bins: int = 2048, min_action: int = -1, max_action: int = 1, fast_skip_tokens: int = 128,
    ) -> None:
        self.tokenizer, self.n_bins, self.min_action, self.max_action, self.fast_skip_tokens = tokenizer, bins, min_action, max_action, fast_skip_tokens

        # Create Uniform Bins + Compute Bin Centers
        self.bins = torch.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - self.fast_skip_tokens - (self.n_bins))

    def __call__(self, action):
        action = torch.clamp(action, min=float(self.min_action), max=float(self.max_action))
        discretized_action = torch.bucketize(action, self.bins, right=False)  # indices in [0, n_bins - 1]

        # Handle single element vs. batch
        if len(discretized_action.shape) == 1:
            # return self.tokenizer.decode(list(self.tokenizer.vocab_size - 1 - self.fast_skip_tokens - discretized_action))
            return list(self.tokenizer.vocab_size - 1 - self.fast_skip_tokens - discretized_action)
        else:
            # return self.tokenizer.batch_decode((self.tokenizer.vocab_size - 1 - self.fast_skip_tokens - discretized_action).tolist())
            return (self.tokenizer.vocab_size - 1 - self.fast_skip_tokens - discretized_action).flatten().tolist()

    def decode_token_ids_to_actions(self, action_token_ids):
        discretized_actions = self.tokenizer.vocab_size - 1 - self.fast_skip_tokens - action_token_ids
        discretized_actions = torch.clamp(discretized_actions, min=0, max=self.bin_centers.shape[0] - 1)

        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins
