# This file provides evaluation utilities for DeepThinkVLA policies.
# Author: Cheng Yin
# Date: 2025-09
# Copyright (c) Cheng Yin. All rights reserved.
# See LICENSE file in the project root for license information.

"""Utils for evaluating DeepThinkVLA policies."""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import textwrap
from transformers import GenerationConfig
from torchvision import transforms
from sft.modeling_deepthinkvla import DeepThinkVLA
from dt_datasets.normalize import Unnormalize_Action
from sft.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_MASK, NUM_ACTIONS_CHUNK, ACTION_DIM

# Initialize important constants
THINK_PREFIX = "First output the thinking process in <think></think> tags and then output the final action in <action></action>."
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
DEEPTHINKVLA_IMAGE_SIZE = 224  # Standard image size expected by DeepThinkVLA

# Configure NumPy print settings
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

def binarize_gripper_action(action: np.ndarray) -> np.ndarray:
    # Create a copy to avoid modifying the original
    normalized_action = action.copy()

    # Binarize to -1 or +1
    normalized_action[..., -1] = np.sign(normalized_action[..., -1])

    return normalized_action

def compose_with_sidepanel(np_img, text, panel_width_px=1024, panel_ratio=0.32,
                           margin=16, title="CoT", max_lines=None):
    H, W, C = np_img.shape
    if panel_width_px is None:
        panel_w = max(120, int(W * panel_ratio))
    else:
        panel_w = int(panel_width_px)

    # 创建新画布（左图 + 右侧栏）
    out = Image.new("RGB", (W + panel_w, H), color=(255, 255, 255))
    out.paste(Image.fromarray(np_img), (0, 0))

    draw = ImageDraw.Draw(out)
    # 字体大小随图高自适应
    try:
        base_font = ImageFont.truetype("DejaVuSans.ttf", size=max(14, H // 42))
        title_font = ImageFont.truetype("DejaVuSans.ttf", size=max(16, H // 36))
    except Exception:
        base_font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    # 侧栏绘制起点
    x0 = W + margin
    y0 = margin
    text_area_w = panel_w - 2 * margin

    # 标题
    if title:
        draw.text((x0, y0), title, fill=(0, 0, 0), font=title_font)
        # 标题下划线
        title_w = draw.textlength(title, font=title_font)
        underline_y = y0 + title_font.getbbox("Ay")[3] - title_font.getbbox("Ay")[1] + 6
        draw.line((x0, underline_y, x0 + min(text_area_w, int(title_w)), underline_y), fill=(0, 0, 0), width=2)
        y0 = underline_y + margin

    # 自动换行
    if text is None:
        text = ""
    paragraphs = text.split("\n")
    wrapped_lines = []
    # 粗略估计行宽 -> 控制 wrap 宽度
    sample = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    sample_w = draw.textlength(sample, font=base_font) or 1
    avg_char_w = sample_w / len(sample)
    max_chars = max(8, int(text_area_w / max(avg_char_w, 1)))

    for p in paragraphs:
        wrapped = textwrap.wrap(p, width=max_chars) if p.strip() else [""]
        wrapped_lines.extend(wrapped)

    # 行高
    line_h = int(base_font.getbbox("Ay")[3] - base_font.getbbox("Ay")[1]) + 4
    # 侧栏可容纳的最大行数（若给了 max_lines 就用它；否则按高度自动算）
    if max_lines is None:
        max_lines = max(1, (H - y0 - margin) // line_h)

    # 截断并加省略号
    display_lines = wrapped_lines[:max_lines]
    truncated = len(wrapped_lines) > max_lines
    if truncated and display_lines:
        display_lines[-1] = display_lines[-1].rstrip(" .") + " …"

    # 逐行写字
    y = y0
    for line in display_lines:
        draw.text((x0, y), line, fill=(0, 0, 0), font=base_font)
        y += line_h

    return np.asarray(out)

def get_vla(cfg) -> torch.nn.Module:
    """
    Load and initialize the VLA model from checkpoint.

    Args:
        cfg: Configuration object

    Returns:
        torch.nn.Module: The initialized VLA model
    """

    # Load the model
    vla = DeepThinkVLA.from_pretrained(
        cfg.pretrained_checkpoint,
        torch_dtype=getattr(torch, cfg.compute_dtype),
        attn_implementation = 'sdpa',
    )

    vla.eval()

    vla = vla.to(DEVICE)

    unomrmalize_action = _get_unomrmalize_action(cfg.pretrained_checkpoint)

    return vla, unomrmalize_action


def _get_unomrmalize_action(checkpoint_path: str) -> None:

    dataset_statistics_path = os.path.join(checkpoint_path, "norm_stats.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        for key in norm_stats["action"].keys():
            norm_stats["action"][key] = np.array(norm_stats["action"][key], dtype=np.float64)
        unomrmalize_action = Unnormalize_Action(
                normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
                stats=norm_stats["action"],
                action_mask=ACTION_MASK,
            )
        return unomrmalize_action
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )
        raise NotImplementedError("No norm stats found!")


def resize_image_for_policy(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    assert isinstance(resize_size, (int, tuple)), "resize_size must be int or tuple"
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    img_pil = Image.fromarray(img)
    resize_trans = transforms.Resize(size=resize_size)
    resized_img = resize_trans(img_pil)
    return np.array(resized_img)

def check_image_format(image: Any) -> None:
    """
    Validate input image format.

    Args:
        image: Image to check

    Raises:
        AssertionError: If image format is invalid
    """
    is_numpy_array = isinstance(image, np.ndarray)
    has_correct_shape = len(image.shape) == 3 and image.shape[-1] == 3
    has_correct_dtype = image.dtype == np.uint8

    assert is_numpy_array and has_correct_shape and has_correct_dtype, (
        "Incorrect image format detected! Make sure that the input image is a "
        "numpy array with shape (H, W, 3) and dtype np.uint8!"
    )


def prepare_image_for_vla(image: np.ndarray) -> Image.Image:
    # Validate format
    check_image_format(image)

    # Resize if needed
    if image.shape != (DEEPTHINKVLA_IMAGE_SIZE, DEEPTHINKVLA_IMAGE_SIZE, 3):
        image = resize_image_for_policy(image, DEEPTHINKVLA_IMAGE_SIZE)

    # Convert to PIL image
    pil_image = Image.fromarray(image).convert("RGB")

    return pil_image


def get_vla_action(
    cfg: Any,
    vla: torch.nn.Module,
    unomrmalize_action,
    processor: Any,
    obs: Dict[str, Any],
    task_label: str,
) -> List[np.ndarray]:
    with torch.inference_mode():
        # Process images
        image = (
            [
                prepare_image_for_vla(obs["full_image"]),
                prepare_image_for_vla(obs["wrist_image"]),
            ]
            if cfg.num_images_in_input > 1
            else [prepare_image_for_vla(obs["full_image"])]
        )

        # Build VLA prompt
        if "cot" in cfg.pretrained_checkpoint:
            prompt = processor.tokenizer.additional_special_tokens[0] * len(image) + THINK_PREFIX + f"Task: {task_label.lower()};"
        else:
            prompt = processor.tokenizer.additional_special_tokens[0] * len(image) + f"Task: {task_label.lower()};"

        # Process primary image
        inputs = processor(text = [prompt], images = image, return_tensors="pt").to(DEVICE, dtype=torch.bfloat16)

        # Generate action
        # Standard VLA output (single-image inputs, discrete actions)
        if 'cot' in cfg.pretrained_checkpoint:
            kwargs = {
                "max_new_tokens": cfg.max_new_tokens,
                "do_sample": False,
                "pad_token_id": processor.tokenizer.pad_token_id,
                "bos_token_id" : processor.tokenizer.bos_token_id,
                "eos_token_id" : None,
                "use_cache" : True,
                "num_beams": 1,
                "temperature" : None,
                "top_p" : None,
                "top_k" : None,
            }
            generation_config = GenerationConfig(**kwargs)
            normalized_actions, input_cot_ids = vla.predict_cot_action(
                input_ids = inputs["input_ids"],
                pixel_values = inputs["pixel_values"],
                attention_mask = inputs["attention_mask"],
                generation_config = generation_config,
            )
            actions = unomrmalize_action(torch.from_numpy(normalized_actions)).numpy()
            cot_text = processor.tokenizer.decode(input_cot_ids[0, inputs["input_ids"].shape[-1]:-1])
        else:
            actions, _ = vla.predict_action(**inputs, unnorm_key=cfg.unnorm_key, do_sample=False)

    # Return action chunk as list of actions
    return [actions[i] for i in range(len(actions))], cot_text

def get_vla_action_mask_cot(
    cfg: Any,
    vla: torch.nn.Module,
    unomrmalize_action,
    processor: Any,
    obs: Dict[str, Any],
    task_label: str,
) -> List[np.ndarray]:
    with torch.inference_mode():
        # Process images
        image = (
            [
                prepare_image_for_vla(obs["full_image"]),
                prepare_image_for_vla(obs["wrist_image"]),
            ]
            if cfg.num_images_in_input > 1
            else [prepare_image_for_vla(obs["full_image"])]
        )

        # Build VLA prompt
        if "cot" in cfg.pretrained_checkpoint:
            prompt = processor.tokenizer.additional_special_tokens[0] * len(image) + THINK_PREFIX + f"Task: {task_label.lower()};"
        else:
            prompt = processor.tokenizer.additional_special_tokens[0] * len(image) + f"Task: {task_label.lower()};"

        # Process primary image
        inputs = processor(text = [prompt], images = image, return_tensors="pt").to(DEVICE, dtype=torch.bfloat16)

        # Generate action
        # Standard VLA output (single-image inputs, discrete actions)
        input_ids = torch.cat([inputs["input_ids"], torch.tensor([[257153, 257154, 257155]], device = inputs["input_ids"].device)], dim=-1)
        attention_mask = torch.cat([inputs["attention_mask"], torch.tensor([[1, 1, 1]], device = inputs["attention_mask"].device)], dim=-1)
        logits, action_start_idx = vla.prompt_cot_predict_action(
            input_cot_ids = input_ids,
            pixel_values = inputs["pixel_values"],
            attention_mask = attention_mask,
        )
        start_indices = action_start_idx.unsqueeze(1)  # [batch_size, 1]
        position_offsets = torch.arange(ACTION_DIM * NUM_ACTIONS_CHUNK, device=logits.device).unsqueeze(0)  # [1, seq_length]
        seq_indices = start_indices + position_offsets  # [batch_size, ACTION_DIM*NUM_ACTIONS_CHUNK]

        # Discrete token-based prediction
        predicted_action_token_ids = (vla.config.action_token_end_idx - vla.config.action_token_begin_idx) - (
            logits[
                torch.arange(logits.shape[0], device=logits.device).unsqueeze(-1),
                seq_indices,
                vla.config.action_token_begin_idx:vla.config.action_token_end_idx + 1
            ]
            .argmax(dim=-1)
            .cpu()
            .numpy()
        )
        discretized_actions = discretized_actions = np.clip(predicted_action_token_ids, a_min=0, a_max=vla.bin_centers.shape[0] - 1)
        normalized_actions = vla.bin_centers[discretized_actions]
        normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        actions = unomrmalize_action(torch.from_numpy(normalized_actions)).numpy()
        cot_text = '<think></think>'


    # Return action chunk as list of actions
    return [actions[i] for i in range(len(actions))], cot_text


# def get_vla_action_mask_cot_random(
#     cfg: Any,
#     vla: torch.nn.Module,
#     unomrmalize_action,
#     processor: Any,
#     obs: Dict[str, Any],
#     task_label: str,
# ) -> List[np.ndarray]:
#     with torch.inference_mode():
#         # Process images
#         image = (
#             [
#                 prepare_image_for_vla(obs["full_image"]),
#                 prepare_image_for_vla(obs["wrist_image"]),
#             ]
#             if cfg.num_images_in_input > 1
#             else [prepare_image_for_vla(obs["full_image"])]
#         )

#         # Build VLA prompt
#         if "cot" in cfg.pretrained_checkpoint:
#             prompt = processor.tokenizer.additional_special_tokens[0] * len(image) + THINK_PREFIX + f"Task: {task_label.lower()};"
#         else:
#             prompt = processor.tokenizer.additional_special_tokens[0] * len(image) + f"Task: {task_label.lower()};"

#         # Process primary image
#         inputs = processor(text = [prompt], images = image, return_tensors="pt").to(DEVICE, dtype=torch.bfloat16)

#         # Generate action
#         # Standard VLA output (single-image inputs, discrete actions)
#         kwargs = {
#             "max_new_tokens": cfg.max_new_tokens,
#             "do_sample": False,
#             "pad_token_id": processor.tokenizer.pad_token_id,
#             "bos_token_id" : processor.tokenizer.bos_token_id,
#             "eos_token_id" : None,
#             "use_cache" : True,
#             "num_beams": 1,
#             "temperature" : None,
#             "top_p" : None,
#             "top_k" : None,
#         }
#         generation_config = GenerationConfig(**kwargs)
#         input_cot_ids = vla.generate(
#             input_ids = inputs["input_ids"],
#             pixel_values = inputs["pixel_values"],
#             attention_mask = inputs["attention_mask"],
#             generation_config = generation_config,
#             stopping_criteria=vla.stopping,
#             logits_processor=vla.proc,
#         )
#         # orig_think_text = processor.tokenizer.decode(input_cot_ids[0, inputs["input_ids"].shape[-1]:])
#         cot_ids_remove_pre_end = input_cot_ids[0, inputs["input_ids"].shape[-1]:][1:-2]
#         cot_ids_remove_pre_end_random_ids = torch.randperm(cot_ids_remove_pre_end.size(0))
#         random_cot_ids = torch.cat([torch.tensor([257153],device=cot_ids_remove_pre_end.device), cot_ids_remove_pre_end[cot_ids_remove_pre_end_random_ids], torch.tensor([257154, 257155], device=cot_ids_remove_pre_end.device)], dim=0).unsqueeze(0)
#         random_think_text = processor.tokenizer.decode(random_cot_ids[0])
#         random_input_cot_ids = torch.cat([inputs["input_ids"], random_cot_ids], dim=-1)

#         logits, action_start_idx = vla.prompt_cot_predict_action(
#             input_cot_ids = random_input_cot_ids,
#             pixel_values = inputs["pixel_values"],
#             attention_mask = torch.ones_like(random_input_cot_ids, device=random_input_cot_ids.device),
#         )
#         start_indices = action_start_idx.unsqueeze(1)  # [batch_size, 1]
#         position_offsets = torch.arange(ACTION_DIM * NUM_ACTIONS_CHUNK, device=logits.device).unsqueeze(0)  # [1, seq_length]
#         seq_indices = start_indices + position_offsets  # [batch_size, ACTION_DIM*NUM_ACTIONS_CHUNK]

#         # Discrete token-based prediction
#         predicted_action_token_ids = (vla.config.action_token_end_idx - vla.config.action_token_begin_idx) - (
#             logits[
#                 torch.arange(logits.shape[0], device=logits.device).unsqueeze(-1),
#                 seq_indices,
#                 vla.config.action_token_begin_idx:vla.config.action_token_end_idx + 1
#             ]
#             .argmax(dim=-1)
#             .cpu()
#             .numpy()
#         )
#         discretized_actions = discretized_actions = np.clip(predicted_action_token_ids, a_min=0, a_max=vla.bin_centers.shape[0] - 1)
#         normalized_actions = vla.bin_centers[discretized_actions]
#         normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

#         actions = unomrmalize_action(torch.from_numpy(normalized_actions)).numpy()

#     # Return action chunk as list of actions
#     return [actions[i] for i in range(len(actions))], random_think_text

def get_vla_action_mask_cot_random(
    cfg: Any,
    vla: torch.nn.Module,
    unomrmalize_action,
    processor: Any,
    obs: Dict[str, Any],
    task_label: str,
) -> List[np.ndarray]:
    with torch.inference_mode():
        # Process images
        image = (
            [
                prepare_image_for_vla(obs["full_image"]),
                prepare_image_for_vla(obs["wrist_image"]),
            ]
            if cfg.num_images_in_input > 1
            else [prepare_image_for_vla(obs["full_image"])]
        )

        # Build VLA prompt
        if "cot" in cfg.pretrained_checkpoint:
            prompt = processor.tokenizer.additional_special_tokens[0] * len(image) + THINK_PREFIX + f"Task: {task_label.lower()};"
        else:
            prompt = processor.tokenizer.additional_special_tokens[0] * len(image) + f"Task: {task_label.lower()};"

        # Process primary image
        inputs = processor(text = [prompt], images = image, return_tensors="pt").to(DEVICE, dtype=torch.bfloat16)

        # Generate action
        # Standard VLA output (single-image inputs, discrete actions)
        input_cot_ids = torch.cat([inputs["input_ids"], torch.tensor([[257153]], device = inputs["input_ids"].device),torch.randint(0, 220000, (1,128), device = inputs["input_ids"].device), torch.tensor([[257154, 257155]], device = inputs["input_ids"].device)], dim=-1)
        attention_mask = torch.ones_like(input_cot_ids, device=input_cot_ids.device)
        logits, action_start_idx = vla.prompt_cot_predict_action(
            input_cot_ids = input_cot_ids,
            pixel_values = inputs["pixel_values"],
            attention_mask = attention_mask,
        )
        start_indices = action_start_idx.unsqueeze(1)  # [batch_size, 1]
        position_offsets = torch.arange(ACTION_DIM * NUM_ACTIONS_CHUNK, device=logits.device).unsqueeze(0)  # [1, seq_length]
        seq_indices = start_indices + position_offsets  # [batch_size, ACTION_DIM*NUM_ACTIONS_CHUNK]

        # Discrete token-based prediction
        predicted_action_token_ids = (vla.config.action_token_end_idx - vla.config.action_token_begin_idx) - (
            logits[
                torch.arange(logits.shape[0], device=logits.device).unsqueeze(-1),
                seq_indices,
                vla.config.action_token_begin_idx:vla.config.action_token_end_idx + 1
            ]
            .argmax(dim=-1)
            .cpu()
            .numpy()
        )
        discretized_actions = discretized_actions = np.clip(predicted_action_token_ids, a_min=0, a_max=vla.bin_centers.shape[0] - 1)
        normalized_actions = vla.bin_centers[discretized_actions]
        normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        actions = unomrmalize_action(torch.from_numpy(normalized_actions)).numpy()
        cot_text = processor.tokenizer.decode(input_cot_ids[0, inputs["input_ids"].shape[-1]:-1])


    # Return action chunk as list of actions
    return [actions[i] for i in range(len(actions))], cot_text