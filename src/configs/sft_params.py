# This file defines configuration dataclasses for DeepThinkVLA.
# Author: Cheng Yin
# Date: 2025-09
# Copyright (c) Cheng Yin. All rights reserved.
# See LICENSE file in the project root for license information.

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from torchvision import transforms

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    base_model_path: Optional[str] = field(default="your_model_path/deepthinkvla_base")
    num_images_in_input: int = field(default=2)
    fast_skip_tokens: int = field(default=128)


@dataclass
class TrainingArgument(TrainingArguments):
    ##########################################################################################
    # Dataloader parameters
    dataloader_num_workers: int = field(default=32)
    """Number of workers for data loading."""

    dataloader_pin_memory: bool = field(default=True)
    """Whether to pin memory for data loading."""

    dataloader_persistent_workers: bool = field(default=True)
    """Whether to use persistent workers for data loading."""

    ##########################################################################################
    # Training parameters
    per_device_train_batch_size: int = field(default=2)
    """Batch size per GPU for training."""

    auto_find_batch_size: bool = field(default=False)
    """Whether to automatically find the batch size."""

    max_steps: int = field(default=3)
    """Total number of training steps to perform. If > 0, overrides num_train_epochs."""

    num_train_epochs: int = field(default=1)
    """Total number of training epochs to perform."""

    deepspeed: str = field(default="./configs/zero2.json")
    """Path to deepspeed config file."""

    gradient_checkpointing: bool = field(default=False)
    """Whether to use gradient checkpointing. Save memory but slower training."""

    bf16: bool = field(default=True)
    """Whether to use bf16. Requires PyTorch >= 1.10 and NVIDIA A100 GPUs."""

    fp16: bool = field(default=False)
    """Whether to use fp16. Requires NVIDIA apex or PyTorch >= 1.6 and NVIDIA GPUs. when using bf16, set to False"""

    tf32: bool = field(default=True)
    """Whether to use tf32. Requires PyTorch >= 1.6 and NVIDIA A100 GPUs."""

    gradient_accumulation_steps: int = field(default=1)
    """Number of updates steps to accumulate before performing a backward/update pass."""

    seed: int = field(default=429)
    """Random seed for reproducibility."""

    max_grad_norm: float = field(default=1.0)
    """Max gradient norm. Used to clip gradients."""

    do_train: bool = field(default=True)
    """Whether to run training."""
    
    ##########################################################################################
    # Save parameters
    save_strategy: str = field(default="steps")
    """The checkpoint save strategy to use."""

    save_steps: float = field(default=2)
    """Number of updates steps before two checkpoint saves. If save_strategy is 'epoch', this is the number of epochs."""

    resume: bool = field(default=False)
    """Whether to resume training from the last checkpoint."""

    save_total_limit: Optional[int] = field(default=40)
    """Limit the total amount of checkpoints. Deletes the older checkpoints in the output_dir."""

    save_safetensors : bool = field(default=True)
    """Whether to save the model as a safetensors file. If False, saves as a pytorch file."""
    
    ##########################################################################################
    # Optimizer parameters
    learning_rate: float = field(default=2.5e-5)
    """Learning rate for training."""

    vision_lr: Optional[float] = field(default=5e-6)
    """Learning rate for vision tower. If None, will use the same learning rate as the rest of the model."""

    merger_lr: Optional[float] = field(default=2.5e-6)
    """Learning rate for merger. If None, will use the same learning rate as the rest of the model."""

    weight_decay: float = field(default=1e-10)
    """Weight decay for AdamW optimizer."""

    # warmup_ratio: float = field(default=0.015)
    """Ratio of total training steps to perform learning rate warmup for. See the documentation for more details."""

    warmup_steps: int = field(default=1000)
    """Number of steps to perform learning rate warmup for. If warmup_ratio is set, this will be ignored."""

    optim: str = field(default="adamw_torch")
    """The optimizer to use. See the documentation for more details."""

    adam_beta1: float = field(default=0.9)
    """Beta1 for AdamW optimizer."""

    adam_beta2: float = field(default=0.95)
    """Beta2 for AdamW optimizer."""

    adam_epsilon: float = field(default=1e-8)
    """Epsilon for AdamW optimizer."""

    lr_scheduler_type: str = field(default="cosine")
    """The scheduler to use. See the documentation for more details."""

    lr_scheduler_kwargs: dict = field(default_factory=lambda: {"min_lr": 2.5e-6})
    
    ##########################################################################################
    # Logging parameters
    output_dir: str = field(default="./checkpoints/deepthinkvla/libero_cot")
    """Directory to save model checkpoints."""
    
    report_to: str = "wandb"
    """The integration to report the results and logs to. Supported platforms are `"tensorboard"` and `"wandb"`."""
    
    run_name: str = field(default="libero_cot")
    """The name of the run. Used for logging and saving checkpoints."""

    logging_steps: int = field(default=1)
    """Number of updates steps before logging training metrics."""

    log_level: str = field(default="info") 
    """Logging level to use. 'debug', training recommend :'info'. 'warning', 'error' and 'critical', plus a 'passive' """

    ##########################################################################################
    # evaluation parameters
    # evaluation_strategy: str = field(default="steps")
    # """The evaluation strategy to use. See the documentation for more details."""

    # eval_steps: int = field(default=1)
    # """Number of updates steps before two evaluation. If eval_strategy is 'epoch', this is the number of epochs."""

    # per_device_eval_batch_size: int = field(default=1)
    # """Batch size per GPU for evaluation."""

    # bf16_full_eval: bool = field(default=True)
    # """Whether to use bf16 for evaluation. Requires PyTorch >= 1.10 and NVIDIA A100 GPUs."""

    # fp16_full_eval: bool = field(default=False)
    # """Whether to use fp16 for evaluation. Requires NVIDIA apex or PyTorch >= 1.6 and NVIDIA GPUs. when using bf16_full_eval, set to False"""

    # prediction_loss_only: bool = field(default=True)
    # """Whether to return only the loss during evaluation. If False, returns the loss and logits."""

    ##########################################################################################
    # LORA arguments
    lora_enable: bool = False
    """Whether to use LoRA for training. If True, will use LoRA for training."""

    vision_lora: bool = False
    """Whether to use LoRA for vision tower. If True, will use LoRA for vision tower."""

    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    """List of namespan to exclude for LoRA. If None, will not exclude any namespan."""

    lora_rank: int = field(default=64)
    """Rank for LoRA. The higher the rank, the more parameters will be trained. If 0, will not use LoRA."""

    lora_alpha: int = field(default=16)
    """Alpha for LoRA. The higher the alpha, the more parameters will be trained. If 0, will not use LoRA."""

    lora_dropout: float = field(default=0.05)
    """Dropout for LoRA. The higher the dropout, the more parameters will be trained. If 0, will not use LoRA."""

    lora_bias: str = field(default="none")
    """Bias for LoRA. The higher the bias, the more parameters will be trained. If 0, will not use LoRA."""

    num_lora_modules: int = field(default=-1)
    """Number of LoRA modules to use. If -1, will use all LoRA modules. If 0, will not use LoRA."""

    freeze_llm: bool = field(default=False)
    """Whether to freeze the LLM. If True, will freeze the LLM."""

    freeze_vision_tower: bool = field(default=False)
    """Whether to freeze the vision tower. If True, will freeze the vision tower."""

    freeze_merger: bool = field(default=False)
    """Whether to freeze the merger. If True, will freeze the merger."""

    ##########################################################################################
    # Quantization parameters
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    """How many bits to use. 16, 8, or 4. If 16, will use fp16. If 8, will use bnb quantization. If 4, will use bnb quantization."""

    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    """Compress the quantization statistics through double quantization. If True, will use double quantization."""

    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    """Quantization data type to use. Should be one of `fp4` or `nf4`. If None, will use the default quantization type."""

    ##########################################################################################
    # Other parameters
    remove_unused_columns: bool = field(default=False)
    """Whether to remove unused columns from the dataset. If False, will keep all columns."""

    # ddp_find_unused_parameters: bool = field(default=False)
    # """Whether to find unused parameters in DDP. If False, will not find unused parameters."""
    # ddp_bucket_cap_mb: int = field(default=100)
    # """The bucket size in MB for DDP. If 0, will use the default bucket size."""
    # torch_compile_model: str = field(default=None)
    # """Whether to use torch compile model. If None, will not use torch compile model."""
      


@dataclass
class DataArguments:
    repo_id: str = field(default = "your_data_repo_id/libero_cot")
    root: Path = field(default=Path("data/datasets/your_dataset_path/libero_cot"))
    download_videos: bool = field(default=True)
    episodes: list[int] | None = field(default=None)
    image_transforms = transforms.Resize(size=(224, 224))
    revision: str | None = field(default=None)
    use_imagenet_stats: bool = field(default=True)
    download_videos: bool = field(default=True)
    video_backend: str = field(default="pyav")
    # other parameters
    reward_delta_indices = None
    action_delta_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    observation_delta_indices = None
    reasoning_dropout: float = field(default=0.0)
    
