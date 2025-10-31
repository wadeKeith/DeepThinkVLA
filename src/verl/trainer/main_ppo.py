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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
import os
import json
import hydra
import ray
from verl.single_controller.ray import RayWorkerGroup
from verl.workers.fsdp_workers import RobActorRolloutRefWorker
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.trainer.ppo.ray_trainer import RayTrainer
import re
import torch
import copy
import random
import numpy as np
from verl import DataProto


class RobRewardManager():
    """The reward manager.
    """
    # TODO: we are requiring a reward manager to be much more stronger than this. so this is fully refactored!
    def __init__(self, num_examine, config, tokenizer) -> None:
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.config=config
        self.tokenizer = tokenizer
        self.pattern = r"<think>\s*([\s\S]*?)\s*</think><action>"
        self.use_cot = "cot" in self.config.actor_rollout_ref.model.path
        self.prompt_end_token_id = [235289, 108]

    def verify(self, data):
        completes = data.batch['complete'].tolist()
        batch_size = data.batch['responses'].size(0)
        assert len(completes) == batch_size
        reward_metrics = {}
        acc_reward = [float(item) for item in completes]
        data.batch['acc'] = torch.tensor(acc_reward, dtype=torch.float32, device=data.batch['responses'].device)
        reward_metrics['acc'] = data.batch['acc'].mean().item()

        if self.use_cot:
            format_reward = []
            for input_cot_ids in data.batch['input_ids']:
                prompt_end_index = ((input_cot_ids.unfold(dimension=1, size=len(self.prompt_end_token_id), step=1) == torch.tensor(self.prompt_end_token_id, device=input_cot_ids.device)).all(dim=-1)).float().argmax(dim=-1) + len(self.prompt_end_token_id)
                cot_texts = self.tokenizer.batch_decode([input_cot_ids[i, prompt_end_index[i]:].tolist() for i in range(len(input_cot_ids))])
                format_match = [re.search(self.pattern, cot_text, re.DOTALL) is not None for cot_text in cot_texts]
                format_reward.append(torch.tensor(format_match).float().mean().item())
            data.batch['format_correctness'] = torch.tensor(format_reward, dtype=torch.float32, device=data.batch['responses'].device)
            reward_metrics['format_correctness'] = data.batch['format_correctness'].mean().item()
            reward_metrics['all'] = (data.batch['acc'] + data.batch['format_correctness']).mean().item()
        else:
            reward_metrics['all'] = data.batch['acc'].mean().item()

        return reward_metrics

    def __call__(self, data: DataProto):
        # aggregate all available reward tensors
        reward_metrics={}
        acc_scores = copy.deepcopy(self.config.verifier.acc_coef * data.batch['acc']).unsqueeze(-1).unsqueeze(-1).expand(-1, data.batch["input_ids"].shape[1], self.config.actor_rollout_ref.model.action_chunks_len * self.config.actor_rollout_ref.model.action_dim_len)
        acc_scores = acc_scores / (data.batch["input_ids"].shape[1] * self.config.actor_rollout_ref.model.action_chunks_len * self.config.actor_rollout_ref.model.action_dim_len)
        if self.use_cot:
            format_reward = []
            for input_cot_ids in data.batch['input_ids']:
                prompt_end_index = ((input_cot_ids.unfold(dimension=1, size=len(self.prompt_end_token_id), step=1) == torch.tensor(self.prompt_end_token_id, device=input_cot_ids.device)).all(dim=-1)).float().argmax(dim=-1) + len(self.prompt_end_token_id)
                cot_texts = self.tokenizer.batch_decode([input_cot_ids[i, prompt_end_index[i]:].tolist() for i in range(len(input_cot_ids))])
                format_match = [re.search(self.pattern, cot_text, re.DOTALL) is not None for cot_text in cot_texts]
                format_reward.append(format_match)
            format_scores = self.config.verifier.format_coef * (torch.tensor(format_reward, dtype=torch.float32, device=data.batch['input_ids'].device).unsqueeze(-1).expand(-1, -1, self.config.data.max_response_length))
            format_scores = format_scores / (data.batch["input_ids"].shape[1] * self.config.data.max_response_length)
            token_level_scores = torch.cat([format_scores, acc_scores], dim=-1)
            token_level_scores = token_level_scores.reshape(token_level_scores.size(0), -1)
        else:
            token_level_scores = acc_scores
            token_level_scores = token_level_scores.reshape(token_level_scores.size(0), -1)
        reward_metrics['acc'] = data.batch['acc'].mean().item()
        if self.use_cot:
            reward_metrics['format_correctness'] = data.batch['format_correctness'].mean().item()
            reward_metrics['all'] = (data.batch['acc'] + data.batch['format_correctness']).mean().item()
        else:
            reward_metrics['all'] = data.batch['acc'].mean().item()

        return token_level_scores, reward_metrics

def set_seed_everywhere(seed: int) -> None:
    """
    Set random seed for all random number generators for reproducibility.

    Args:
        seed: The random seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        if os.path.isfile(str(config.trainer.runtime_env)):
            with open(str(config.trainer.runtime_env), 'r') as f:
                runtime_env = json.load(f)
            print(runtime_env)
            ray.init(runtime_env=runtime_env, num_cpus=config.ray_init.num_cpus)
        else:
            ray.init(
                runtime_env={
                    "env_vars": {
                        "TOKENIZERS_PARALLELISM": "true",
                        "NCCL_DEBUG": "WARN",
                        "VLLM_LOGGING_LEVEL": "WARN",
                        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                    }
                },
                num_cpus=config.ray_init.num_cpus,
            )
        
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))
    # create a timeline trace file to analyze the performance
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # instantiate tokenizer
        from verl.utils import hf_processor

        processor = hf_processor(local_path, trust_remote_code=False)  # used for multimodal LLM, could be none

        # vllm early verify
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(RobActorRolloutRefWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
        }

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(RobActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = RobRewardManager( num_examine=0, config=config, tokenizer=processor.tokenizer) # note: verifier is called both inside reward_fn and outside.
        val_reward_fn = RobRewardManager( num_examine=1,config=config, tokenizer=processor.tokenizer)
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = RayTrainer(
            config=config,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            device_name=config.trainer.device,
        )
        trainer.init_workers()
        trainer.fit()



if __name__ == "__main__":
    set_seed_everywhere(429)
    main()
