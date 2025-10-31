from typing import List, Optional, Tuple, Union
import torch
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaCausalLMOutputWithPast,
    PaliGemmaForConditionalGeneration,
    add_start_docstrings_to_model_forward,
    PALIGEMMA_INPUTS_DOCSTRING,
    Cache,
    StaticCache,
    HybridCache,
    PaliGemmaConfig
)
from transformers.utils import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Iterable
from torch import nn
from transformers import StoppingCriteria, LogitsProcessor, StoppingCriteriaList, LogitsProcessorList
import numpy as np
from sft.constants import NUM_ACTIONS_CHUNK, ACTION_DIM

logger = logging.get_logger(__name__)

class SeqEosTokenCriteria(StoppingCriteria):
    def __init__(self, eos_token_id: Union[int, List[int]]):
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = torch.tensor(eos_token_id,  dtype=torch.long)

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        # input_ids: (batch_size, seq_len)
        t = self.eos_token_id.numel()
        if input_ids.shape[1] < t:
            return torch.full((input_ids.shape[0],), False, device=input_ids.device, dtype=torch.bool)

        tail = input_ids[:, -t:]                                  # (B, t)
        target = self.eos_token_id.to(tail.device).unsqueeze(0)           # (1, t)
        matches_per_sample = torch.all(tail == target, dim=-1)    # (B,)
        return matches_per_sample.to(device=input_ids.device, dtype=torch.bool)

class TailVocabMaskProcessor(LogitsProcessor):
    def __init__(self, vocab_size: int, ban_start_ids:int, ban_end_ids: int, allowed_ids: Optional[Iterable[int]] = None):
        assert ban_end_ids - ban_start_ids > 0 and ban_end_ids <= (vocab_size - 1)
        self.vocab_size = vocab_size
        self.allowed = set(allowed_ids or [])

        banned = torch.zeros(vocab_size, dtype=torch.bool)

        banned[ban_start_ids:ban_end_ids] = True
        for tid in self.allowed:
            if 0 <= tid < vocab_size:
                banned[tid] = False
        self.registered_mask = banned  # CPU 上的模板
        self.registered_mask[1] = True  # 禁用 <eos> token

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores: (batch*beams, vocab_size)
        banned = self.registered_mask.to(scores.device)
        # 用 dtype 的最小值屏蔽（float32/-3.4e38, float16/-65504）
        scores = scores.masked_fill(banned, torch.finfo(scores.dtype).min)
        return scores

def get_actions_mask_cot(labels, action_token_begin_idx, action_token_end_idx, ignore_index):
    # Create a tensor marking positions of IGNORE_INDEX
    newline_positions = labels != ignore_index

    # Extract the action part only
    action_tokens_only_mask = (labels >= action_token_begin_idx) & (labels <= action_token_end_idx)

    useful_action_mask = action_tokens_only_mask * newline_positions

    # Calculate cumulative sum to identify regions between newlines
    cumsum = torch.cumsum(useful_action_mask, dim=1)

    # Create the mask
    mask = 1 <= cumsum

    return mask * action_tokens_only_mask


class DeepThinkVLA(PaliGemmaForConditionalGeneration):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__(config)
        self.bins = np.linspace(-1, 1, 2048)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.stopping = StoppingCriteriaList([SeqEosTokenCriteria([self.config.think_end_token_index, self.config.action_start_token_index])])
        self.proc = LogitsProcessorList([TailVocabMaskProcessor(vocab_size = self.config.text_config.vocab_size,
                                                                ban_start_ids = self.config.action_token_begin_idx,
                                                                ban_end_ids= self.config.action_token_end_idx + 1)])
        self.prompt_end_token_id = [235289, 108]
    @add_start_docstrings_to_model_forward(PALIGEMMA_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=PaliGemmaCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: int = 0,
        cot_length = None,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        is_training = token_type_ids is not None and labels is not None

        ########################################################################################################################
        # add in NAR + Bi-Attn
        if cot_length is not None:
            # input_ids: [Pad, Prompt, CoT]
            # Get number of tokens in prompt (excluding the start token)
            action_start_idx = input_ids.ne(self.pad_token_id).sum(dim=1) - 1

            # Create fake labels tensor (needed for action mask)
            labels = input_ids.clone()
            labels[:] = self.config.ignore_index

            ##############################################################################################################################################
            # self._prepare_input_for_action_prediction(input_ids, attention_mask)
            ##############################################################################################################################################
            placeholder_action_token_ids = torch.full(
                (input_ids.shape[0], ACTION_DIM * NUM_ACTIONS_CHUNK), 
                fill_value=2, 
                device=input_ids.device, 
                dtype=input_ids.dtype
            )
            # Add stop token to sequence (needed in non-causal bi-directional self-attention, as it appears at train time)
            action_end_id = torch.full(
                (input_ids.shape[0], 1), 
                fill_value=self.config.action_end_token_index, 
                device=input_ids.device, 
                dtype=input_ids.dtype
            )
            stop_token_id = torch.full(
                (input_ids.shape[0], 1), 
                fill_value=self.config.eos_token_id, 
                device=input_ids.device, 
                dtype=input_ids.dtype
            )
            input_ids = torch.cat([input_ids, placeholder_action_token_ids, action_end_id, stop_token_id], dim=-1)

            # Extend the attention mask to fit the new shape of input
            # Note: Only batch size == 1 supported right now
            mask_extension = torch.ones(
                (attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]),
                device=attention_mask.device,
                dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)
            ##############################################################################################################################################
            ##############################################################################################################################################

            ##############################################################################################################################################
            # self._prepare_labels_for_action_prediction(labels, input_ids)
            ##############################################################################################################################################
            labels_extension = torch.full(
                (labels.shape[0], input_ids.shape[-1] - labels.shape[-1]), 
                fill_value=self.config.action_token_begin_idx,
                device=labels.device, 
                dtype=labels.dtype
            )
            labels = torch.cat([labels, labels_extension], dim=-1)

            # Replace last label token with stop token
            labels[:, -1] = self.config.eos_token_id
            labels[:, -2] = self.config.action_end_token_index
            ##############################################################################################################################################
            ##############################################################################################################################################

            # [Pad, Prompt, CoT, Action_placeholder, Stop]
            sorted_indices = torch.argsort(((input_ids.ne(self.pad_token_id))).int(), dim=1, descending=True, stable=True)
            input_ids = torch.gather(input_ids, 1, sorted_indices)
            attention_mask = torch.gather(attention_mask, 1, sorted_indices)
            labels = torch.gather(labels, 1, sorted_indices)

            # [Prompt, CoT, Action_placeholder, Stop, Pad]
            inputs_embeds = self.get_input_embeddings()(input_ids)
            all_actions_mask = get_actions_mask_cot(labels = labels,
                                                        action_token_begin_idx = self.config.action_token_begin_idx,
                                                        action_token_end_idx = self.config.action_token_end_idx,
                                                        ignore_index = self.config.ignore_index).unsqueeze(-1)
            inputs_embeds = inputs_embeds * ~all_actions_mask

            cache_position = torch.arange(
                0, 0 + inputs_embeds.shape[1], device=inputs_embeds.device
            )

            position_ids = cache_position.unsqueeze(0) + 1  # Paligemma positions are 1-indexed

            image_features = self.get_image_features(pixel_values)

            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            if inputs_embeds[special_image_mask].numel() != image_features.numel():
                image_tokens_in_text = torch.sum(input_ids == self.config.image_token_index)
                raise ValueError(
                    f"Number of images does not match number of special image tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                    "tokens from image embeddings."
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

            cot_action_start_idx = ((input_ids.unfold(dimension=1, size=len(self.prompt_end_token_id), step=1) == torch.tensor(self.prompt_end_token_id, device=input_ids.device)).all(dim=-1)).float().argmax(dim=-1) + len(self.prompt_end_token_id)
            cot_action_mask = (torch.arange(input_ids.shape[1], device=input_ids.device).view(1, input_ids.shape[1])>=cot_action_start_idx.view(-1, 1).long())
            token_type_ids = torch.ones_like(input_ids)
            token_type_ids = torch.where(cot_action_mask, token_type_ids, 0) & attention_mask
            ########################################################################################################################################
            causal_mask = self._update_causal_mask(
                attention_mask, token_type_ids, None, cache_position, input_ids, inputs_embeds, True, action_start_idx
            )

            language_model_output = self.language_model(
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                cache_position=cache_position,
            )

            action_start_indices = action_start_idx.unsqueeze(1)  # [batch_size, 1]
            action_position_offsets = torch.arange(ACTION_DIM * NUM_ACTIONS_CHUNK, device=language_model_output.logits.device).unsqueeze(0)  # [1, seq_length]
            action_seq_indices = action_start_indices + action_position_offsets  # [batch_size, ACTION_DIM*NUM_ACTIONS_CHUNK]

            cot_start_indices = (action_start_idx - cot_length).unsqueeze(1)  # [batch_size, 1]
            cot_position_offsets = torch.arange(cot_length, device=language_model_output.logits.device).unsqueeze(0)  # [1, cot_length]
            cot_seq_indices = cot_start_indices + cot_position_offsets  # [batch_size, cot_length]

            cot_logits = language_model_output.logits[
                torch.arange(language_model_output.logits.shape[0], device=language_model_output.logits.device).unsqueeze(-1),  
                cot_seq_indices, 
                :
            ]
            action_logits = language_model_output.logits[
                torch.arange(language_model_output.logits.shape[0], device=language_model_output.logits.device).unsqueeze(-1),  
                action_seq_indices, 
                :
            ]
            return cot_logits, action_logits
        ########################################################################################################################

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        ########################################################################################################################
        # add in NAR + Bi-Attn
        if is_training:
            all_actions_mask = get_actions_mask_cot(labels = labels,
                                                    action_token_begin_idx = self.config.action_token_begin_idx,
                                                    action_token_end_idx = self.config.action_token_end_idx,
                                                    ignore_index = self.config.ignore_index).unsqueeze(-1)
            inputs_embeds = inputs_embeds * ~all_actions_mask
        ########################################################################################################################

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0) + 1  # Paligemma positions are 1-indexed

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            if inputs_embeds[special_image_mask].numel() != image_features.numel():
                image_tokens_in_text = torch.sum(input_ids == self.config.image_token_index)
                raise ValueError(
                    f"Number of images does not match number of special image tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                    "tokens from image embeddings."
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # mask out pad-token-ids in labels for BC
        if labels is not None and self.pad_token_id in labels:
            logger.warning_once(
                "`labels` contains `pad_token_id` which will be masked with `config.ignore_index`. "
                "You have to mask out `pad_token_id` when preparing `labels`, this behavior will be removed in v.4.46.",
            )
            labels = torch.where(input_ids == self.pad_token_id, self.config.ignore_index, labels)

        ########################################################################################################################
        # add in NAR + Bi-Attn
        if is_training:
            action_start_idx = torch.where((input_ids == self.config.action_start_token_index), torch.arange(input_ids.shape[-1], device=input_ids.device), -1).max(dim=1).values
        else:
            action_start_idx = None
        ########################################################################################################################
        causal_mask = self._update_causal_mask(
            attention_mask, token_type_ids, past_key_values, cache_position, input_ids, inputs_embeds, is_training, action_start_idx
        )
        outputs = self.language_model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        logits = outputs.logits
        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()

            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return PaliGemmaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )
    def _update_causal_mask(
        self,
        attention_mask,
        token_type_ids,
        past_key_values,
        cache_position,
        input_ids=None,
        inputs_embeds=None,
        is_training: bool = False,
        action_start_idx = None,
    ):
        if self.config.text_config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        using_static_cache = isinstance(past_key_values, StaticCache)
        min_dtype = torch.finfo(self.dtype).min
        inputs_lead_dim = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        sequence_length = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        elif isinstance(past_key_values, HybridCache):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else cache_position[0] + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            return attention_mask

        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=self.dtype, device=cache_position.device
        )
        # Causal diagonal mask only if training, otherwise attend to the whole prefix. Training-specific attn for prefix is handled below
        if sequence_length != 1:
            if is_training:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            else:
                causal_mask[:, :sequence_length] = 0.0

        causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_lead_dim, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
            # we are training thus we need to create a full mask on the image + prefix but causal on suffix
            if is_training:
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    token_type_ids[:, None, None, :].to(causal_mask.device) == 0, 0
                )
        ########################################################################################################################
        # add in NAR + Bi-Attn
        if action_start_idx is not None: # added for COT support
            last_row = causal_mask[:, :, -1:, :].clone()
            cot_mask = torch.arange(causal_mask.shape[-2], device=causal_mask.device).view(1, 1, causal_mask.shape[-2], 1) >= action_start_idx.view(causal_mask.shape[0], 1, 1, 1)
            new_mask = torch.where(cot_mask, last_row, causal_mask)
            causal_mask = new_mask
        ########################################################################################################################
        return causal_mask

    def _prepare_labels_for_action_prediction(self, labels, input_ids):
        labels_extension = torch.full(
            (labels.shape[0], input_ids.shape[-1] - labels.shape[-1]), 
            fill_value=self.config.action_token_begin_idx,
            device=labels.device, 
            dtype=labels.dtype
        )
        labels = torch.cat([labels, labels_extension], dim=-1)

        # Replace last label token with stop token
        labels[:, -1] = self.config.eos_token_id
        labels[:, -2] = self.config.action_end_token_index

        return labels

    def _prepare_input_for_action_prediction(self, input_ids, attention_mask):
        """Prepares input for action prediction by adding necessary tokens"""
        # Add (ACTION_DIM * NUM_ACTIONS_CHUNK) placeholder tokens to input_ids to simulate action tokens
        placeholder_action_token_ids = torch.full(
            (input_ids.shape[0], ACTION_DIM * NUM_ACTIONS_CHUNK), 
            fill_value=2, 
            device=input_ids.device, 
            dtype=input_ids.dtype
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        # Add stop token to sequence (needed in non-causal bi-directional self-attention, as it appears at train time)
        action_end_id = torch.full(
            (input_ids.shape[0], 1), 
            fill_value=self.config.action_end_token_index, 
            device=input_ids.device, 
            dtype=input_ids.dtype
        )
        stop_token_id = torch.full(
            (input_ids.shape[0], 1), 
            fill_value=self.config.eos_token_id, 
            device=input_ids.device, 
            dtype=input_ids.dtype
        )
        input_ids = torch.cat([input_ids, action_end_id, stop_token_id], dim=-1)

        # Extend the attention mask to fit the new shape of input
        # Note: Only batch size == 1 supported right now
        mask_extension = torch.ones(
            (attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]),
            device=attention_mask.device,
            dtype=attention_mask.dtype
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        return input_ids, attention_mask

    def prompt_cot_predict_action(
        self,
        input_cot_ids,
        pixel_values,
        attention_mask,
    ):
        # input_cot_ids: [Pad, Prompt, CoT]
        action_start_idx = input_cot_ids.ne(self.pad_token_id).sum(dim=1) - 1
        ########################################################################################################################################
        # Create fake labels tensor (needed for action mask)
        labels = input_cot_ids.clone()
        labels[:] = self.config.ignore_index

        # Prepare inputs by adding necessary tokens
        input_cot_ids, attention_mask = self._prepare_input_for_action_prediction(input_cot_ids, attention_mask)

        # Update labels tensor for action mask computation later
        labels = self._prepare_labels_for_action_prediction(labels, input_cot_ids)
        # [Pad, Prompt, CoT, Action_placeholder, Stop]
        sorted_indices = torch.argsort(((input_cot_ids.ne(self.pad_token_id))).int(), dim=1, descending=True, stable=True)
        input_cot_ids = torch.gather(input_cot_ids, 1, sorted_indices)
        attention_mask = torch.gather(attention_mask, 1, sorted_indices)
        labels = torch.gather(labels, 1, sorted_indices)

        # [Prompt, CoT, Action_placeholder, Stop, Pad]
        inputs_embeds = self.get_input_embeddings()(input_cot_ids)
        all_actions_mask = get_actions_mask_cot(labels = labels,
                                                    action_token_begin_idx = self.config.action_token_begin_idx,
                                                    action_token_end_idx = self.config.action_token_end_idx,
                                                    ignore_index = self.config.ignore_index).unsqueeze(-1)
        inputs_embeds = inputs_embeds * ~all_actions_mask

        cache_position = torch.arange(
            0, 0 + inputs_embeds.shape[1], device=inputs_embeds.device
        )

        position_ids = cache_position.unsqueeze(0) + 1  # Paligemma positions are 1-indexed

        image_features = self.get_image_features(pixel_values)

        special_image_mask = (input_cot_ids == self.config.image_token_index).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            image_tokens_in_text = torch.sum(input_cot_ids == self.config.image_token_index)
            raise ValueError(
                f"Number of images does not match number of special image tokens in the input text. "
                f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                "tokens from image embeddings."
            )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        cot_action_start_idx = ((input_cot_ids.unfold(dimension=1, size=len(self.prompt_end_token_id), step=1) == torch.tensor(self.prompt_end_token_id, device=input_cot_ids.device)).all(dim=-1)).float().argmax(dim=-1) + len(self.prompt_end_token_id)
        cot_action_mask = (torch.arange(input_cot_ids.shape[1], device=input_cot_ids.device).view(1, input_cot_ids.shape[1])>=cot_action_start_idx.view(-1, 1).long())
        token_type_ids = torch.ones_like(input_cot_ids)
        token_type_ids = torch.where(cot_action_mask, token_type_ids, 0) & attention_mask
        ########################################################################################################################################
        causal_mask = self._update_causal_mask(
            attention_mask, token_type_ids, None, cache_position, input_cot_ids, inputs_embeds, True, action_start_idx
        )

        outputs = self.language_model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            cache_position=cache_position,
        )

        return outputs.logits, action_start_idx

    def predict_cot_action(
        self,
        input_ids,
        pixel_values,
        attention_mask,
        generation_config = None,
    ):
        # input_ids: [Pad, Prompt]
        ########################################################################################################################################
        # COT Generation
        ########################################################################################################################################
        input_cot_ids = super().generate(
            input_ids = input_ids,
            pixel_values = pixel_values,
            attention_mask = attention_mask,
            generation_config = generation_config,
            stopping_criteria=self.stopping,
            logits_processor=self.proc,
        )
        # input_cot_ids: [Pad, Prompt, CoT, Pad]
        attention_mask = input_cot_ids.ne(self.pad_token_id).int()
        sorted_indices = torch.argsort((~(input_cot_ids.ne(self.pad_token_id))).int(), dim = 1, descending=True, stable=True)
        input_cot_ids = torch.gather(input_cot_ids, 1, sorted_indices)
        attention_mask = torch.gather(attention_mask, 1, sorted_indices)
        ########################################################################################################################################
        # Action Generation
        ########################################################################################################################################
        # input_cot_ids: [Pad, Prompt, CoT]
        logits, action_start_idx = self.prompt_cot_predict_action(
            input_cot_ids = input_cot_ids,
            pixel_values = pixel_values,
            attention_mask = attention_mask,
        )

        start_indices = action_start_idx.unsqueeze(1)  # [batch_size, 1]
        position_offsets = torch.arange(ACTION_DIM * NUM_ACTIONS_CHUNK, device=logits.device).unsqueeze(0)  # [1, seq_length]
        seq_indices = start_indices + position_offsets  # [batch_size, ACTION_DIM*NUM_ACTIONS_CHUNK]

        # Discrete token-based prediction
        predicted_action_token_ids = (self.config.action_token_end_idx - self.config.action_token_begin_idx) - (
            logits[
                torch.arange(logits.shape[0], device=logits.device).unsqueeze(-1),
                seq_indices,
                self.config.action_token_begin_idx:self.config.action_token_end_idx + 1
            ]
            .argmax(dim=-1)
            .cpu()
            .numpy()
        )
        discretized_actions = discretized_actions = np.clip(predicted_action_token_ids, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]
        normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        return normalized_actions, input_cot_ids

    def generate_action_verl(
        self,
        input_ids,
        pixel_values,
        attention_mask,
        do_sample = True,
        temperature = None,
        generation_config = None,
    ):
        # input_ids: [Pad, Prompt]
        ########################################################################################################################################
        # COT Generation
        ########################################################################################################################################
        input_cot_ids = super().generate(
            input_ids = input_ids,
            pixel_values = pixel_values,
            attention_mask = attention_mask,
            generation_config = generation_config,
            stopping_criteria=self.stopping,
            logits_processor=self.proc,
        )

        # input_cot_ids: [Pad, Prompt, CoT, Pad]
        attention_mask = input_cot_ids.ne(self.pad_token_id).int()
        sorted_indices = torch.argsort((~(input_cot_ids.ne(self.pad_token_id))).int(), dim = 1, descending=True, stable=True)
        input_cot_ids = torch.gather(input_cot_ids, 1, sorted_indices)
        attention_mask = torch.gather(attention_mask, 1, sorted_indices)

        return_input_cot_ids = input_cot_ids.clone()
        return_attention_mask = attention_mask.clone()
        ########################################################################################################################################
        # Action Generation
        ########################################################################################################################################
        # input_cot_ids: [Pad, Prompt, CoT]
        logits, action_start_idx = self.prompt_cot_predict_action(
            input_cot_ids = input_cot_ids,
            pixel_values = pixel_values,
            attention_mask = attention_mask,
        )

        start_indices = action_start_idx.unsqueeze(1)  # [batch_size, 1]
        position_offsets = torch.arange(ACTION_DIM * NUM_ACTIONS_CHUNK, device=logits.device).unsqueeze(0)  # [1, seq_length]
        seq_indices = start_indices + position_offsets  # [batch_size, ACTION_DIM*NUM_ACTIONS_CHUNK]

        if do_sample == False:
            predicted_action_token_ids = (self.config.action_token_end_idx - self.config.action_token_begin_idx) - (
                logits[
                    torch.arange(logits.shape[0], device=logits.device).unsqueeze(-1),
                    seq_indices,
                    self.config.action_token_begin_idx:self.config.action_token_end_idx + 1
                ]
                .argmax(dim=-1)
            )
        else:
            assert temperature>0, "Please provide temperature when using sampling!"
            action_logits = logits[
                torch.arange(logits.shape[0], device=logits.device).unsqueeze(-1),
                seq_indices, 
                self.config.action_token_begin_idx:self.config.action_token_end_idx + 1
            ]
            scaled_logits = action_logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            assert probs.shape[-1] == 2048
            probs_flat = probs.reshape(-1, probs.shape[-1])
            sampled_indices_flat = torch.multinomial(probs_flat, num_samples=1)
            predicted_action_token_ids = (
                (self.config.action_token_end_idx - self.config.action_token_begin_idx)
                - sampled_indices_flat
            ).view(action_logits.shape[0], -1)
        discretized_actions = np.clip(predicted_action_token_ids.cpu().numpy(), a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]
        normalized_actions = normalized_actions.reshape(-1, ACTION_DIM)

        return (
            normalized_actions,
            (self.config.action_token_end_idx - self.config.action_token_begin_idx)
            - predicted_action_token_ids
            + self.config.action_token_begin_idx,
            return_input_cot_ids,
            return_attention_mask,
        )
