# This file orchestrates supervised fine-tuning for DeepThinkVLA.
# Author: Cheng Yin
# Date: 2025-09
# Copyright (c) Cheng Yin. All rights reserved.
# See LICENSE file in the project root for license information.

from pathlib import Path
import os
from sft.sft_runner import TrainRunner
from transformers import HfArgumentParser, BitsAndBytesConfig
from transformers import AutoProcessor
from configs.sft_params import DataArguments, ModelArguments, TrainingArgument
import torch
from dt_datasets.dataset import LiberoDataset, PadDataCollator
from dt_datasets.action_tokenizer import ActionTokenizer
from transformers.utils import logging
import warnings
import ast
from peft import LoraConfig, get_peft_model
from sft.utils import (
    find_target_linear_names,
    configure_vision_tower,
    configure_llm,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    safe_save_model_for_hf_trainer,
)
from sft.modeling_deepthinkvla import DeepThinkVLA

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.get_logger(__name__)


local_rank = None


def rank0_print(*args):
    if local_rank == 0 or local_rank == "0" or local_rank is None:
        print(*args)


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArgument,
):

    global local_rank

    ##########################################################################################
    # parameters check
    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert (
            not training_args.vision_lora
        ), "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."

    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError(
            "If `vision_lora` is True, `freeze_vision_tower` must also be True."
        )

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(
                training_args.lora_namespan_exclude
            )
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["vision_tower"]

    ##########################################################################################
    # load the model
    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["vision_tower"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                ),
            )
        )

    model = DeepThinkVLA.from_pretrained(
        model_args.base_model_path,
        torch_dtype=compute_dtype,
        attn_implementation = 'sdpa',
        **bnb_model_from_pretrained_args
    )

    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(
        model_to_configure, training_args
    )

    ##########################################################################################
    # Quantization and LoRA
    if training_args.bits in [4, 8]:
        model.config.torch_dtype = (
            torch.float32
            if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": True},
        )

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(
                model,
                lora_namespan_exclude=lora_namespan_exclude,
                num_lora_modules=training_args.num_lora_modules,
            ),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)

            if "lm_head" in name or "embed_token" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    ##########################################################################################
    # Create the train dataset and sample dataset
    processor = AutoProcessor.from_pretrained(model_args.base_model_path)

    action_tokenizer = ActionTokenizer(processor.tokenizer, fast_skip_tokens=model_args.fast_skip_tokens)
    train_dataset = LiberoDataset(
        data_args=data_args,
        processor = processor,
        action_tokenizer = action_tokenizer,
        use_wrist_image = model_args.num_images_in_input > 1,
        dataset_flag="train",
    )

    # data collator
    datacollator = PadDataCollator(processor.tokenizer, model.config.ignore_index)

    # 2.2 run experiment
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        data_collator=datacollator,
        resume_from_checkpoint=training_args.resume,
        processor=processor,
        action_tokenizer = action_tokenizer,
    )

    # 2.3 run experiment
    experiment.train()

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=False
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_state_dict.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(
            experiment.trainer, output_dir=training_args.output_dir
        )


if __name__ == "__main__":
    # os.environ["WANDB_PROJECT"] = "deepthinkvla"
    # os.environ["WANDB_MODE"] = "offline"
    # os.environ["WANDB_NAME"] = "libero_cot"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArgument))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    train(model_args, data_args, training_args)
