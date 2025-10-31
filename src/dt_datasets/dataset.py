# This file implements the LIBERO dataset wrapper for DeepThinkVLA.
# Author: Cheng Yin
# Date: 2025-09
# Copyright (c) Cheng Yin. All rights reserved.
# See LICENSE file in the project root for license information.

from torchvision import transforms 
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset,LeRobotDatasetMetadata
import torch
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollatorMixin
from configs.sft_params import DataArguments
import json
import numpy as np
from dt_datasets.normalize import Normalize_Action
import torchvision.transforms as T
from sft.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_MASK
from transformers import AutoTokenizer
import random
from torch.nn.utils.rnn import pad_sequence


NON_PREFIX = ""
THINK_PREFIX = "First output the thinking process in <think></think> tags and then output the final action in <action></action>."

def create_reasoning_tokens(
    reasoning: torch.Tensor,
    tokenizer: AutoTokenizer,
    dropout: float,
):
    if torch.all(reasoning == tokenizer.pad_token_id):
        return NON_PREFIX, "<think></think>"

    reasoning_text = tokenizer.decode(reasoning, skip_special_tokens=True)

    has_think_tags = "<think>" in reasoning_text and "</think>" in reasoning_text

    keep_reasoning = has_think_tags and random.random() >= dropout

    prefix = THINK_PREFIX if keep_reasoning else NON_PREFIX
    output_reasoning_text = reasoning_text if keep_reasoning else "<think></think>"

    return prefix, output_reasoning_text


IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}

Tensor2PIL_transform = T.ToPILImage()

def resolve_delta_timestamps(data_args:DataArguments, ds_meta:LeRobotDatasetMetadata):
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == "next.reward" and data_args.reward_delta_indices is not None:
            delta_timestamps[key] = [
                i / ds_meta.fps for i in data_args.reward_delta_indices
            ]
        if key == "action" and data_args.action_delta_indices is not None:
            delta_timestamps[key] = [
                i / ds_meta.fps for i in data_args.action_delta_indices
            ]
        if (
            key.startswith("observation.")
            and data_args.observation_delta_indices is not None
        ):
            delta_timestamps[key] = [
                i / ds_meta.fps for i in data_args.observation_delta_indices
            ]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(data_args: DataArguments):
    ds_meta = LeRobotDatasetMetadata(
        data_args.repo_id,
        root=data_args.root,
        revision=data_args.revision,
    )
    delta_timestamps = resolve_delta_timestamps(data_args, ds_meta)
    dataset = LeRobotDataset(
        data_args.repo_id,
        root=data_args.root,
        episodes=data_args.episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=data_args.image_transforms,
        revision=data_args.revision,
        download_videos=data_args.download_videos,
        video_backend=data_args.video_backend,
    )

    if data_args.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(
                    stats, dtype=torch.float32
                )
    state_action_stats = json.load(open(data_args.root / "meta/norm_stats.json"))
    state_stats = state_action_stats["state"]
    action_stats = state_action_stats["action"]
    for key in state_stats.keys():
        dataset.meta.stats["observation.state"][key] = (
            np.array(
                state_stats[key],
                dtype=dataset.meta.stats["observation.state"][key].dtype,
                like=dataset.meta.stats["observation.state"][key],
            )
            if key in dataset.meta.stats["observation.state"]
            else np.array(state_stats[key], dtype=np.float64)
        )
        dataset.meta.stats["action"][key] = (
            np.array(
                action_stats[key],
                dtype=dataset.meta.stats["action"][key].dtype,
                like=dataset.meta.stats["action"][key],
            )
            if key in dataset.meta.stats["action"]
            else np.array(action_stats[key], dtype=np.float64)
        )

    return dataset


class LiberoDataset(Dataset):
    def __init__(
        self,
        data_args: DataArguments,
        processor,
        action_tokenizer,
        use_wrist_image,
        dataset_flag="train",
    ):
        self.data_args = data_args
        self.processor = processor
        self.action_tokenizer = action_tokenizer
        self.use_wrist_image = use_wrist_image
        self.dataset = make_dataset(data_args)
        self.dataset_flag = dataset_flag
        self.normalize_action = Normalize_Action(
            ACTION_PROPRIO_NORMALIZATION_TYPE,
            self.dataset.meta.stats["action"],
            ACTION_MASK,
        )
        self.set_epoch(0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.dataset_flag == "train":
            
            height, width = item['observation.images.image'].shape[-2], item['observation.images.image'].shape[-1]
            Data_augmentation_transforms_pipeline = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        size=(int(height * 0.95), int(width * 0.95)),
                        scale=(0.9, 0.9),
                        ratio=(1.0, 1.0),
                    ),  # Equivalent to RandomCrop
                    transforms.Resize((height, width)),
                    transforms.RandomRotation(
                        5
                    ),  # Random rotation in the range (-5, 5)
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.4, saturation=0.5
                    ),
                ]
            )

            item['observation.images.image'] = Data_augmentation_transforms_pipeline(item['observation.images.image'])

        item['action'] = self.normalize_action(item['action'])

        image = (
            [
                Tensor2PIL_transform(item["observation.images.image"]),
                Tensor2PIL_transform(item["observation.images.wrist_image"]),
            ]
            if self.use_wrist_image
            else [Tensor2PIL_transform(item["observation.images.image"])]
        )

        prefix_reasoning_text, reasoning_text = create_reasoning_tokens(item['reasoning'], self.action_tokenizer.tokenizer, dropout=self.data_args.reasoning_dropout)
        cleaned = item['task'].lower().strip().replace("_", " ")
        prefix_text = self.processor.tokenizer.additional_special_tokens[0] * len(image) + prefix_reasoning_text + f"Task: {cleaned};"

        action_tokens = self.action_tokenizer(item["action"])
        cot_text = reasoning_text + "<action>"
        item_input = self.processor(
            text=[prefix_text],
            images=image,
            suffix = [cot_text],
            return_tensors="pt",
        )
        input_ids = item_input["input_ids"][:, :-1]
        token_type_ids = item_input["token_type_ids"][:, :-1]
        attention_mask = item_input["attention_mask"][:, :-1]
        labels = item_input['labels'][:, :-1]
        pixel_values = item_input["pixel_values"]

        input_ids = torch.cat(
            [
                input_ids,
                torch.tensor([action_tokens], dtype=input_ids.dtype),
                torch.full(
                    size=[input_ids.shape[0], 1],
                    fill_value=257156,
                    dtype=input_ids.dtype,
                ),
                torch.full(
                    size=[input_ids.shape[0], 1],
                    fill_value=self.processor.tokenizer.eos_token_id,
                    dtype=input_ids.dtype,
                ),
            ],
            dim=-1,
        )
        token_type_ids = torch.cat(
            [
                token_type_ids,
                torch.full(
                    size=[token_type_ids.shape[0], len(action_tokens) + 2],
                    fill_value=1,
                    dtype=token_type_ids.dtype,
                ),
            ],
            dim=-1,
        )
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.full(
                    size=[attention_mask.shape[0], len(action_tokens) + 2],
                    fill_value=1,
                    dtype=attention_mask.dtype,
                ),
            ],
            dim=-1,
        )
        labels = torch.cat(
            [
                labels,
                torch.tensor([action_tokens], dtype=labels.dtype),
                torch.full(
                    size=[labels.shape[0], 1],
                    fill_value=257156,
                    dtype=labels.dtype,
                ),
                torch.full(
                    size=[labels.shape[0], 1],
                    fill_value=self.processor.tokenizer.eos_token_id,
                    dtype=labels.dtype,
                ),
            ],
            dim=-1,
        )
        return {"input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "pixel_values": pixel_values}

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch

class PadDataCollator(DataCollatorMixin):
    def __init__(self, tokenizer, ignore_index):
        super().__init__()
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
    def __call__(self, features, return_tensors=None):
        input_ids, token_type_ids, labels = tuple([feature[key][0] for feature in features] for key in ("input_ids", "token_type_ids", "labels"))
        pixel_values = [feature["pixel_values"] for feature in features]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=self.ignore_index)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).int()

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        pixel_values = torch.cat(pixel_values, dim=0)

        batch_output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            token_type_ids=token_type_ids,
        )
        return batch_output
