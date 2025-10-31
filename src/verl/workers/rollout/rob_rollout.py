# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single GPU model.
Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model to perform generation.
"""
import contextlib
import torch
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
import verl.utils.torch_functional as verl_F
from .base import BaseRollout

from transformers import GenerationConfig, AutoProcessor

from verl.utils.libero_utils import get_libero_env, get_libero_dummy_action, get_libero_image, get_libero_wrist_image, quat2axisangle, normalize_gripper_action, invert_gripper_action, save_rollout_video
from verl.utils.device import get_device_name, get_device_id, get_torch_device
import numpy as np
from PIL import Image
import tensorflow as tf
from verl import DataProto
from libero.libero import benchmark
from einops import rearrange

import re
import os
import copy
import gc
from multiprocessing import Process, Queue
from collections import defaultdict
import json
from dt_datasets.normalize import Unnormalize_Action
from sft.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_MASK, NUM_ACTIONS_CHUNK, ACTION_DIM

__all__ = ['RobHFRollout']

NON_PREFIX = ""
THINK_PREFIX = "First output the thinking process in <think></think> tags and then output the final action in <action></action>."

def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image

def center_crop_image(image):
    batch_size = 1
    crop_scale = 0.9

    # Convert to TF Tensor and record original data type (should be tf.uint8)
    image = tf.convert_to_tensor(np.array(image))
    orig_dtype = image.dtype

    # Convert to data type tf.float32 and values between [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Crop and then resize back to original size
    image = crop_and_resize(image, crop_scale, batch_size)

    # Convert back to original data type
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    # Convert back to PIL Image
    image = Image.fromarray(image.numpy())
    image = image.convert("RGB")
    return image


def env_worker(task_name, task_id, trial_id, config, input_queue, output_queue, is_valid, max_steps):
    
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    initial_state = initial_states[trial_id]
    
    
    env = None
    while True:
        try:
            env, task_description = get_libero_env(task, resolution=256)
            break  
        except:
            print(f"*** env initialization failed ***")
            if env is not None:
                try:
                    env.close()  
                except Exception as e:
                    print(f"error when close the env: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            print("gc collect finish")
    
    env.reset()
    obs = env.set_init_state(initial_state)
    
    
    t = 0
    valid_images = []
    while t < config.num_steps_wait:
        obs, _, _, _ = env.step(get_libero_dummy_action())
        t += 1
        
    if is_valid:
        img = obs["agentview_image"][::-1, ::-1]
        valid_images.append(img)
    
    output_queue.put({
        'type': 'init',
        'obs': obs,
        "task_description":task_description,
        'valid_images': valid_images.copy(),
        'task_file_name': f"{task_name}_task_{task_id}_trial_{trial_id}",
        'active': True,
        'complete': False,
        'finish_step': 0
    })
    
    active = True
    complete = False
    finish_step = 0
    
    while True:
        
        action = input_queue.get()
        if action is None:
            env.close()
            output_queue.put({'type': 'terminate'})
            break
        
        
        step_images = []
        for i in range(len(action)):
            a = action[i]
            # normalized_action = normalize_gripper_action(a, binarize=True)
            # inverted_action = invert_gripper_action(normalized_action)
            obs, reward, done, info = env.step(a.tolist())
            
            if is_valid:
                img = obs["agentview_image"][::-1, ::-1]
                step_images.append(img)
            
            
            finish_step += 1
            #if done or finish_step >= config.max_steps[config.task_suite_name]:
            if done or finish_step >= max_steps:
                active = False
                complete = done
                break
        
        
        output_data = {
            'type': 'step',
            'obs': obs,
            'active': active,
            'complete': complete,
            'finish_step': finish_step,
            'valid_images': step_images.copy() if is_valid else []
        }
        output_queue.put(output_data)


class RobHFRollout(BaseRollout):

    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module
        self.max_steps = {   "libero_spatial": 480,   # max step length 193
                                    "libero_object": 480,    # max step length 254
                                    "libero_goal": 480,      # max step length 270
                                    "libero_10": 480,        # max step length 505
                                    "libero_90": 480         # max step length 373 org 400 now change to 512
                                }
        self.processor = AutoProcessor.from_pretrained(config.pretrained_checkpoint)

        dataset_statistics_path = os.path.join(config.pretrained_checkpoint, "norm_stats.json")
        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path, "r") as f:
                norm_stats = json.load(f)
            for key in norm_stats["action"].keys():
                norm_stats["action"][key] = np.array(norm_stats["action"][key], dtype=np.float64)
            self.unomrmalize_action = Unnormalize_Action(
                normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
                stats=norm_stats["action"],
                action_mask=ACTION_MASK,
            )
        else:
            print(
                "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
                "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
                "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
            )
            raise NotImplementedError("No norm stats found!")
        self.use_cot = "cot" in self.config.pretrained_checkpoint

    def process_input(self,inputs:list, task_descriptions:list):
        images = []
        prompts = []
        for i in range(len(inputs)):
            task_description = task_descriptions[i]

            if self.config.center_crop:
                images.append(center_crop_image(Image.fromarray(inputs[i]['observation.images.image']).convert("RGB")))
                if self.config.num_images_in_input > 1:
                    images.append(center_crop_image(Image.fromarray(inputs[i]['observation.images.wrist_image']).convert("RGB")))
            else:
                images.append(Image.fromarray(inputs[i]['observation.images.image']).convert("RGB"))
                if self.config.num_images_in_input > 1:
                    images.append(Image.fromarray(inputs[i]['observation.images.wrist_image']).convert("RGB"))

            cleaned = task_description.lower().strip().replace("_", " ")
            if self.use_cot:
                prompt = self.processor.tokenizer.additional_special_tokens[0] * self.config.num_images_in_input + THINK_PREFIX + f"Task: {cleaned};"
            else:
                prompt = self.processor.tokenizer.additional_special_tokens[0] * self.config.num_images_in_input + NON_PREFIX + f"Task: {cleaned};"
            prompts.append(prompt)

        batchdata = self.processor(
            text=prompts,
            images=images,
            padding=True,
            return_tensors="pt",
            padding_side="left",
            add_special_tokens=False,
        )
        return batchdata

    def generate_sequences(self, prompts):
        batch_size = prompts.batch.batch_size[0]

        if prompts.meta_info.get('n_samples') is None:
            micro_batch_size = self.config.val_micro_batch_size if self.config.val_micro_batch_size is not None else 1
        else:
            micro_batch_size = self.config.get('micro_batch_size', batch_size)

        num_chunks = max(batch_size // micro_batch_size, 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    def _generate_minibatch(self, prompts):
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get('n_samples', 1)
        task_id = prompts.batch['task_id'].repeat_interleave(n_samples, dim=0)
        trial_id = prompts.batch['trial_id'].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(prompts.non_tensor_batch['task_suite_name'], n_samples)
        max_steps = self.max_steps[self.config.task_suite_name]
        batch_size = task_id.size(0)
        is_valid = meta_info.get('n_samples') is None
        global_steps = meta_info.get('global_steps', 0) if is_valid else 0

        if is_valid:
            # do_sample==False -> greedy decoding
            kwargs = {
                "max_new_tokens": 2048,
                "do_sample": False,
                "pad_token_id":self.processor.tokenizer.pad_token_id,
                "bos_token_id" : self.processor.tokenizer.bos_token_id,
                "eos_token_id" : None,
                "use_cache" : True,
                "num_beams": 1,
                "temperature" : None,
                "top_p" : None,
                "top_k" : None,
            }
        else:
            # do_sample -> use rollout config
            kwargs = {
                "max_new_tokens": self.config.response_length,
                "do_sample": True,
                "pad_token_id":self.processor.tokenizer.pad_token_id,
                "bos_token_id" : self.processor.tokenizer.bos_token_id,
                "eos_token_id" : None,
                "use_cache" : True,
                "num_beams": 1,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "temperature": self.config.cot_temperature,
            }

        # make config according to generate mode
        generation_config = GenerationConfig(**kwargs)

        processes = []
        input_queues = []
        output_queues = []

        for idx in range(batch_size):
            task_name = task_suite_name[idx]
            t_id = task_id[idx][0].item()
            tr_id = trial_id[idx][0].item()
            input_q = Queue()
            output_q = Queue()
            p = Process(
                target=env_worker,
                args=(task_name, t_id, tr_id, self.config, input_q, output_q, is_valid, max_steps)
            )
            p.start()
            processes.append(p)
            input_queues.append(input_q)
            output_queues.append(output_q)

        inputs = []
        task_descriptions = []
        task_records = []
        valid_video = defaultdict(list)
        for idx in range(batch_size):
            init_data = output_queues[idx].get(timeout=600)
            assert init_data['type'] == 'init'
            task_descriptions.append(init_data["task_description"])
            inputs.append(self._obs_to_input(init_data['obs']))
            task_records.append({
                "active": init_data['active'],
                "complete": init_data['complete'],
                "finish_step": init_data['finish_step'],
                "task_file_name": init_data['task_file_name']
            })
            if is_valid:
                valid_video[init_data['task_file_name']].extend(init_data['valid_images'])

        step = 0
        vla_history = []
        while step < max_steps:
            active_indices = [i for i, r in enumerate(task_records) if r['active']]

            current_inputs = inputs
            current_task_descriptions = task_descriptions

            vla_input = self.process_input(current_inputs, current_task_descriptions)
            vla_input.update(meta_info)
            vla_output = self._generate_one_step(vla_input, generation_config)
            actions = vla_output["action"]

            step_data = {
                    "responses": vla_output["responses"],
                    "input_ids": vla_output["input_ids"],
                    "attention_mask": vla_output["attention_mask"],
                    "pixel_values": vla_output["pixel_values"],
                    "action": actions,
                    "step": step
                }
            vla_history.append(step_data)

            for idx in active_indices:
                input_queues[idx].put(actions[idx])

            new_inputs = inputs.copy()
            for idx in active_indices:
                result = output_queues[idx].get(timeout=600)
                assert result['type'] == 'step'
                new_inputs[idx] = self._obs_to_input(result['obs'])
                task_records[idx]['active'] = result['active']
                task_records[idx]['complete'] = result['complete']
                task_records[idx]['finish_step'] = result['finish_step']
                if is_valid:
                    valid_video[task_records[idx]['task_file_name']].extend(result['valid_images'])

            inputs = new_inputs
            step += self.config.action_chunks_len

        for q in input_queues:
            q.put(None)
        for p in processes:
            p.join(timeout=600)
            if p.is_alive():
                p.terminate()

        get_torch_device().empty_cache()

        if is_valid:
            for task_file, images in valid_video.items():
                complete = any(r['complete'] for r in task_records if r['task_file_name'] == task_file)
                save_rollout_video(
                    images,
                    self.config.experiment_name,
                    task_file,
                    global_steps,
                    complete
                )

        self.module.train()

        batch = {
                'responses': [],
                'input_ids': [],  # here input_ids become the whole sentences
                'attention_mask': [],
                'pixel_values': []
            }
        for k in ["responses", "input_ids", "attention_mask", "pixel_values"]:
            for h in vla_history:
                batch[k].append(h[k])

        for k,v in batch.items():
            batch[k] = torch.stack(v,dim=1) 

        batch["complete"] = []
        batch["finish_step"] = []

        for k in task_records:
            batch["complete"].append(k["complete"])
            batch["finish_step"].append(k["finish_step"])

        batch["complete"] = torch.tensor(batch["complete"], dtype=torch.bool, device=batch['responses'].device)
        batch["finish_step"] = torch.tensor(batch["finish_step"], dtype=torch.int64, device=batch['responses'].device)

        output_batch = TensorDict(
            batch,
            batch_size=batch_size)
        return DataProto(batch=output_batch)

    @torch.no_grad()
    def _generate_one_step(self, prompts: dict, generation_config):
        input_ids = prompts['input_ids'].to(get_device_id()) # (bs, prompt_length)
        attention_mask = prompts['attention_mask'].to(get_device_id())  # left-padded attention_mask
        pixel_values = prompts["pixel_values"].to(get_device_id())

        param_ctx = contextlib.nullcontext()

        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)

        with param_ctx, torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            normalized_actions, action_response, return_input_cot_ids, return_attention_mask = self.module.generate_action_verl(
                input_ids = input_ids,
                pixel_values = pixel_values,
                attention_mask = attention_mask, 
                do_sample=generation_config.do_sample,
                temperature = self.config.action_temperature if generation_config.do_sample else None,
                generation_config = generation_config
            )
            actions = self.unomrmalize_action(torch.from_numpy(normalized_actions)).numpy()
            actions = actions.reshape(-1 ,NUM_ACTIONS_CHUNK, ACTION_DIM)
        assert self.processor.tokenizer.pad_token_id is not None

        assert return_input_cot_ids.ndim == 2
        return_input_cot_ids = verl_F.pad_sequence_to_length(
            return_input_cot_ids,
            max_seq_len=self.config.max_prompt_length + self.config.response_length,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            left_pad=True,
        )
        assert return_attention_mask.ndim == 2
        return_attention_mask = verl_F.pad_sequence_to_length(
            return_attention_mask,
            max_seq_len=self.config.max_prompt_length + self.config.response_length,
            pad_token_id=0,
            left_pad=True,
        )

        assert pixel_values.ndim == 4
        pixel_values = rearrange(pixel_values, '(b n) c h w -> b n c h w', b=input_ids.shape[0], n=self.config.num_images_in_input)
        assert return_input_cot_ids.device.type == 'cuda'
        assert action_response.device.type == 'cuda'
        assert return_attention_mask.device.type == 'cuda'
        assert pixel_values.device.type == 'cuda'
        batch ={
                'responses': action_response,
                'input_ids': return_input_cot_ids,
                'attention_mask': return_attention_mask,
                "pixel_values":pixel_values,
                "action":actions,
            }

        return batch

    def _obs_to_input(self, obs):

        if self.config.num_images_in_input > 1:
            return {
                "observation.images.image": get_libero_image(obs, 224),
                "observation.images.wrist_image": get_libero_wrist_image(obs, 224),
                "observation.state": np.concatenate([
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"]
                ]),
            }
        else:
            return {
                "observation.images.image": get_libero_image(obs, 224),
                "observation.state": np.concatenate([
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"]
                ])
            }
