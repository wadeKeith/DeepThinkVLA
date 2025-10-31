# patch_paligemma_multimg.py
from vllm.model_executor.models.paligemma import (
    PaliGemmaProcessingInfo,
    PaliGemmaMultiModalProcessor,
    PaliGemmaForConditionalGeneration
)
from collections.abc import Mapping, Sequence
from typing import Optional, Union, List
from vllm.multimodal.parse import ImageProcessorItems, ImageEmbeddingItems, MultiModalDataItems
from vllm.multimodal.processing import PromptUpdate, PromptReplacement, PromptInsertion, PromptUpdateDetails, PromptIndexTargets
from vllm.multimodal.inputs import  MultiModalKwargs
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.sequence import IntermediateTensors
from vllm.model_executor.layers.sampler import SamplerOutput

import torch

def _patch_get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        image_token_id = hf_config.image_token_index

        def get_replacement(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems))

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                )
                
            return [image_token_id] * num_image_tokens

        # Paligemma 1 and 2 have different tokenizer.add_bos_token
        # Insert <image>*n + <bos> after <bos> for Paligemma 1
        # Insert <image>*n + <bos> for Paligemma 2
        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            )
        ]


def patch_get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(torch.where(input_ids == self.config.image_token_index, self.config.pad_token_id, input_ids))
        if multimodal_embeddings is not None:
            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, torch.stack(multimodal_embeddings))
        return inputs_embeds


def patch_get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

def patch_paligemma_vllm():
    PaliGemmaProcessingInfo.get_supported_mm_limits = patch_get_supported_mm_limits

    PaliGemmaForConditionalGeneration.get_input_embeddings = patch_get_input_embeddings

    PaliGemmaMultiModalProcessor._get_prompt_updates = _patch_get_prompt_updates
