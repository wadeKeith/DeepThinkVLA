# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, get_device_id
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, logprobs_from_logits_v2
from verl.workers.actor import BasePPOActor


__all__ = ["RobDataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RobDataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()
        self.prompt_end_token_id = [235289, 108]

    def generate_traj_mask(self, end_step, traj_len):
        """
        Args:
            end_step: (batch_size,), 
            traj_len: 
        Returns:
            mask: (batch_size, traj_len),
        """
        steps = torch.arange(traj_len, device=end_step.device)  # (traj_len,)
        steps_expanded = steps.unsqueeze(0).expand(end_step.size(0), -1)
        mask = steps_expanded < end_step.unsqueeze(1)  # (batch_size, traj_len)
        return mask

    def apply_mask_with_grad_control(self, log_probs, entropy, mask, calculate_entropy=False):
        """
        Args:
            log_probs: (batch_size, traj_len, ...)
            entropy:   (batch_size, traj_len, ...)
            mask:      (batch_size, traj_len, ...)
        Returns:
            log_probs_masked: 
            entropy_masked:   
        """

        log_probs_masked = torch.where(
            mask,
            log_probs,
            torch.zeros_like(log_probs, requires_grad=False)  
        )
        if calculate_entropy:
            entropy_masked = torch.where(
                mask,
                entropy,
                torch.zeros_like(entropy, requires_grad=False)   
            )
        else:
            entropy_masked = None
        return log_probs_masked, entropy_masked

    def _forward_micro_batch(self, micro_batch, cot_temperature, action_temperature, calculate_entropy=False):
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        batch_size = micro_batch['responses'].size(0)
        traj_len = micro_batch['responses'].size(1)
        tot_pad_len = micro_batch['input_ids'].size(2)

        assert all(micro_batch[key].size(0) == batch_size for key in ['responses', 'input_ids', 'attention_mask', 'pixel_values'])
        assert all(micro_batch[key].size(1) == traj_len for key in ['responses', 'input_ids', 'attention_mask', 'pixel_values'])
        assert all(micro_batch[key].size(2) == tot_pad_len for key in [ 'input_ids', 'attention_mask'])
        response_length = micro_batch['responses'].size(-1) # ACTION_DIM * NUM_ACTIONS_CHUNK

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            attention_mask = micro_batch['attention_mask']
            pixel_values = micro_batch["pixel_values"]
            responses = micro_batch["responses"]
            entropy = None

            input_ids = input_ids.reshape((batch_size * traj_len,) + input_ids.shape[2:])
            attention_mask = attention_mask.reshape((batch_size * traj_len,) + attention_mask.shape[2:])
            pixel_values = pixel_values.reshape((batch_size * traj_len,) + pixel_values.shape[2:])
            responses = responses.reshape((batch_size * traj_len,) + responses.shape[2:])

            cot_logits, action_logits = self.actor_module(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    pixel_values=pixel_values.flatten(0, 1),
                                    use_cache=False,  # prevent model thinks we are generating
                                    cot_length=self.config.cot_length,
                                    )
            assert (((input_ids.unfold(dimension=1, size=len(self.prompt_end_token_id), step=1) == torch.tensor(self.prompt_end_token_id, device=input_ids.device)).all(dim=-1)).any(dim=-1)).all(), "Some batch size haven't str 'Out: ' "
            cot_start_index = ((input_ids.unfold(dimension=1, size=len(self.prompt_end_token_id), step=1) == torch.tensor(self.prompt_end_token_id, device=input_ids.device)).all(dim=-1)).float().argmax(dim=-1) + len(self.prompt_end_token_id)
            cot_mask = (torch.arange(input_ids.shape[1], device=input_ids.device).view(1, input_ids.shape[1])>=cot_start_index.view(-1, 1).long())[:, -self.config.cot_length:]

            action_logits = action_logits[..., self.actor_module.config.action_token_begin_idx:self.actor_module.config.action_token_end_idx + 1]  # Shape: [batch_size * traj_len, action_response_len, 2048]
            action_responses = responses - self.actor_module.config.action_token_begin_idx

            cot_banned = torch.zeros(self.actor_module.config.text_config.vocab_size, dtype=torch.bool, device=cot_logits.device)
            cot_banned[self.actor_module.config.action_token_begin_idx:self.actor_module.config.action_token_end_idx + 1] = True
            cot_banned[1] = True  # <eos> token
            cot_banned = cot_banned.expand(cot_logits.shape[0], cot_logits.shape[1], -1).to(cot_logits.device)
            cot_logits = cot_logits.masked_fill(cot_banned, torch.finfo(cot_logits.dtype).min)
            cot_responses = input_ids[:,-self.config.cot_length:]

            action_logits = action_logits.div(action_temperature) 
            cot_logits = cot_logits.div(cot_temperature)
            action_log_probs = logprobs_from_logits_v2(action_logits, action_responses)
            cot_log_probs = logprobs_from_logits_v2(cot_logits, cot_responses)
            if calculate_entropy:
                action_entropy = verl_F.entropy_from_logits(action_logits)  # (bs * traj_len, response_length)
                cot_entropy = verl_F.entropy_from_logits(cot_logits)  # (bs * traj_len, cot_length)

            assert len(action_log_probs.shape)==2 and len(cot_log_probs.shape)==2, f"action_log_probs shape: {action_log_probs.shape}, cot_log_probs shape: {cot_log_probs.shape}"
            action_log_probs = action_log_probs.reshape((batch_size, traj_len * self.config.action_chunks_len, self.config.action_dim_len) )
            cot_log_probs = cot_log_probs.reshape((batch_size, traj_len, self.config.cot_length) )  # (bs, traj_len, cot_length)
            cot_mask = cot_mask.reshape((batch_size, traj_len, self.config.cot_length)) # (bs, traj_len, cot_length)
            if calculate_entropy:
                action_entropy = action_entropy.reshape((batch_size, traj_len,) + action_entropy.shape[1:]) # (bs, traj_len, response_length)
                cot_entropy = cot_entropy.reshape((batch_size, traj_len,) + cot_entropy.shape[1:])  # (bs, traj_len, cot_length)

            traj_mask_action = self.generate_traj_mask(micro_batch['finish_step'], traj_len * self.config.action_chunks_len)
            traj_mask_cot = self.generate_traj_mask(micro_batch['finish_step'] // self.config.action_chunks_len + 1, traj_len)
            cot_mask_all = traj_mask_cot.unsqueeze(-1).expand(-1, -1, self.config.cot_length) & cot_mask # (bs, traj_len, cot_length)
            action_log_probs = torch.where(
                traj_mask_action.unsqueeze(-1),
                action_log_probs,
                torch.zeros_like(action_log_probs, requires_grad=False)  
            )
            cot_log_probs = torch.where(
                cot_mask_all,
                cot_log_probs,
                torch.zeros_like(cot_log_probs, requires_grad=False)  
            )
            action_log_probs = action_log_probs.reshape((batch_size, traj_len, response_length))
            action_mask = traj_mask_action.unsqueeze(-1).expand(-1, -1, self.config.action_dim_len).reshape((batch_size, traj_len, response_length))

            if calculate_entropy:
                action_entropy = torch.where(
                    action_mask,
                    action_entropy,
                    torch.zeros_like(action_entropy, requires_grad=False)  
                )
                cot_entropy = torch.where(
                    cot_mask_all,
                    cot_entropy,
                    torch.zeros_like(cot_entropy, requires_grad=False)  
                )

            log_probs = torch.cat([cot_log_probs, action_log_probs], dim=-1)  # (bs, traj_len, cot_length + response_length)
            mask = torch.cat([cot_mask_all, action_mask], dim=-1)  # (bs, traj_len, cot_length + response_length)
            if calculate_entropy:
                entropy = torch.cat([cot_entropy, action_entropy], dim=-1) # (bs, traj_len, cot_length + response_length)

            log_probs = log_probs.reshape((batch_size, -1))
            mask = mask.reshape((batch_size, -1))
            if calculate_entropy:
                entropy = entropy.reshape((batch_size, -1))
            assert log_probs.shape == mask.shape == entropy.shape, f"log_probs shape: {log_probs.shape}, mask shape: {mask.shape}, entropy shape: {entropy.shape}"
            return entropy, log_probs, mask

    def _forward_micro_batch_update(self, input_ids, attention_mask, pixel_values, responses, cot_temperature, action_temperature, calculate_entropy):
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            cot_logits, action_logits = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values.flatten(0, 1),
                use_cache=False,  # prevent model thinks we are generating
                cot_length=self.config.cot_length,
            )

            assert cot_logits.requires_grad 
            assert action_logits.requires_grad

            action_logits = action_logits[..., self.actor_module.config.action_token_begin_idx:self.actor_module.config.action_token_end_idx + 1]  # Shape: [batch_size * traj_len, action_response_len, 2048]
            action_responses = responses - self.actor_module.config.action_token_begin_idx

            cot_banned = torch.zeros(self.actor_module.config.text_config.vocab_size, dtype=torch.bool, device=cot_logits.device)
            cot_banned[self.actor_module.config.action_token_begin_idx:self.actor_module.config.action_token_end_idx + 1] = True
            cot_banned[1] = True  # <eos> token
            cot_banned = cot_banned.expand(cot_logits.shape[0], cot_logits.shape[1], -1).to(cot_logits.device)
            cot_logits = cot_logits.masked_fill(cot_banned, torch.finfo(cot_logits.dtype).min)
            cot_responses = input_ids[:,-self.config.cot_length:]

            action_logits = action_logits.div(action_temperature) 
            cot_logits = cot_logits.div(cot_temperature)

            action_log_probs = logprobs_from_logits_v2(action_logits, action_responses)
            cot_log_probs = logprobs_from_logits_v2(cot_logits, cot_responses)
            if calculate_entropy:
                action_entropy = verl_F.entropy_from_logits(action_logits)  # (traj_mini_len, response_length)
                cot_entropy = verl_F.entropy_from_logits(cot_logits)  # (traj_mini_len, cot_length)

            log_probs = torch.cat([cot_log_probs, action_log_probs], dim=-1)  # (traj_mini_len, cot_length + response_length)
            log_probs = log_probs.reshape((1, -1))
            entropy = None
            if calculate_entropy:
                entropy = torch.cat([cot_entropy, action_entropy], dim=-1) # (traj_mini_len, cot_length + response_length)
                entropy = entropy.reshape((1, -1))

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False):
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        cot_temperature = data.meta_info["cot_temperature"]
        action_temperature = data.meta_info["action_temperature"]
        self.pad_token_id = data.meta_info['pad_token_id']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'pixel_values',"finish_step"]
        batch = data.select(batch_keys=select_keys).batch

        micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        mask_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, mask = self._forward_micro_batch(
                    micro_batch,
                    cot_temperature=cot_temperature,
                    action_temperature=action_temperature,
                    calculate_entropy=calculate_entropy,
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
            mask_lst.append(mask)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        mask = torch.concat(mask_lst, dim=0)

        return log_probs, entropys, mask

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
        cot_temperature = data.meta_info['cot_temperature']
        action_temperature = data.meta_info['action_temperature']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'pixel_values', 'old_log_probs', 'advantages', "mask"]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        assert self.config.ppo_micro_batch_size_per_gpu == 1

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)
        metrics = {}

        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_device_id()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_device_id())  # actor device is cpu when using offload

                    mask = data["mask"] # [1, traj_length * (cot_length + response_length)]
                    mask_sum = mask.sum(axis=None)

                    old_log_prob = data["old_log_probs"] # [1, traj_length * (cot_length + response_length)]
                    advantages = data["advantages"] # [1, traj_length * (cot_length + response_length)]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    batch_size = data['responses'].size(0)
                    assert batch_size == 1, "Only support ppo_micro_batch_size_per_gpu = 1 for now"
                    traj_length = data['responses'].size(1)

                    input_ids = data['input_ids'] # [1, traj_length, prompt_length + cot_length]
                    attention_mask = data['attention_mask'] # [1, traj_length, prompt_length + cot_length]
                    pixel_values = data["pixel_values"] # [1, traj_length, num_cameras, c, h, w]
                    responses = data["responses"] # [1, traj_length,  responses_length]

                    input_ids = input_ids.reshape((batch_size * traj_length,) + input_ids.shape[2:])
                    attention_mask = attention_mask.reshape((batch_size * traj_length,) + attention_mask.shape[2:])
                    pixel_values = pixel_values.reshape((batch_size * traj_length,) + pixel_values.shape[2:])
                    responses = responses.reshape((batch_size * traj_length,) + responses.shape[2:])

                    loss_info = {
                        #'actor/entropy_loss': entropy_loss.detach().item(),
                        'actor/pg_loss':0,
                        'actor/pg_clipfrac': 0,
                        'actor/ppo_kl': 0,
                        "actor/pg_clipfrac_lower": 0,
                        "actor/kl_loss": 0,
                    }

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True

                    assert traj_length % self.config.traj_mini_batch_size ==0
                    traj_split_num = int(traj_length/self.config.traj_mini_batch_size)
                    for i in range(0, traj_length, int(traj_length / traj_split_num)):
                        entropy, log_prob = self._forward_micro_batch_update(
                            input_ids=input_ids[i : i + int(traj_length / traj_split_num)],
                            attention_mask=attention_mask[i : i + int(traj_length / traj_split_num)],
                            pixel_values=pixel_values[i : i + int(traj_length / traj_split_num)],
                            responses=responses[i : i + int(traj_length / traj_split_num)],
                            cot_temperature=cot_temperature,
                            action_temperature=action_temperature,
                            calculate_entropy=calculate_entropy,
                        )
                        slice_id = i*(self.config.cot_length + self.config.action_dim_len*self.config.action_chunks_len)
                        next_slice_id = (i+int(traj_length/traj_split_num))*(self.config.cot_length + self.config.action_dim_len*self.config.action_chunks_len)

                        old_log_prob_tmp = old_log_prob[:, slice_id: next_slice_id]
                        advantages_tmp = advantages[:, slice_id: next_slice_id]
                        mask_tmp = mask[:, slice_id: next_slice_id]
                        
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob_tmp,
                            log_prob=log_prob,
                            advantages=advantages_tmp,
                            response_mask=mask_tmp,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )
                        mask_tmp_sum = mask_tmp.sum(axis=None)
                        pg_loss = pg_loss* mask_tmp_sum
                        pg_clipfrac = pg_clipfrac* mask_tmp_sum / mask_sum
                        ppo_kl = ppo_kl* mask_tmp_sum / mask_sum
                        
                        if entropy_coeff != 0:
                            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=mask_tmp, loss_agg_mode=loss_agg_mode)

                            # compute policy loss
                            policy_loss = (pg_loss - entropy_loss * entropy_coeff) / mask_sum
                        else:
                            policy_loss = pg_loss / mask_sum
                        
                        if self.config.use_kl_loss:
                            ref_log_prob_tmp = data["ref_log_prob"][:, slice_id: next_slice_id]
                            # compute kl loss
                            kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob_tmp, kl_penalty=self.config.kl_loss_type)
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=mask_tmp, loss_agg_mode=loss_agg_mode)
                            kl_loss = kl_loss* mask_tmp_sum / mask_sum

                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            loss_info["actor/kl_loss"] = loss_info["actor/kl_loss"] + kl_loss.detach().item()
                        
                        loss = policy_loss / self.gradient_accumulation
                        loss.backward()
                        loss_info['actor/pg_loss'] =  loss_info['actor/pg_loss'] + policy_loss.detach().item()
                        loss_info['actor/pg_clipfrac'] = loss_info['actor/pg_clipfrac'] + pg_clipfrac.detach().item()
                        loss_info['actor/ppo_kl'] = loss_info['actor/ppo_kl'] +  ppo_kl.detach().item()
                        loss_info['actor/pg_clipfrac_lower'] = loss_info['actor/pg_clipfrac_lower'] + pg_clipfrac_lower.detach().item()

                    append_to_dict(metrics, loss_info)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
                get_torch_device().empty_cache()
        self.actor_optimizer.zero_grad()
        get_torch_device().synchronize()
        torch.distributed.barrier()
        get_torch_device().empty_cache()
        return metrics
