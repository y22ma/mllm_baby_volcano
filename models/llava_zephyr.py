from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn

from .constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import AutoModel, MistralModel, MistralConfig, MistralForCausalLM, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

cuda_available = torch.cuda.is_available()
device = "cuda" if cuda_available else 'cpu' # the device to load the model onto

class LlavaZephyrModel(MistralForCausalLM):
    def __init__(self, config: MistralConfig):
        super(MistralForCausalLM, self).__init__(config)
        self.config = config
        self.llm = MistralModel(config)
        self.vis_enc = AutoModel.from_pretrained("facebook/dinov2-large")
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
        self.image_projector = nn.Linear(config.vis_enc_hidden_dim, config.projector_hidden_dim)
        self.llm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if images is not None:
            # pass the image tensor throught the image encoder and the projector to get the
            # image tokens, referred to as H in the original LLAVA paper
            img_tokens = self._generate_img_tokens(images)
            inputs_embeds = self._combine_image_text_embeds(input_ids, img_tokens)
            if labels is not None:
                labels = self._combine_image_text_labels(input_ids, img_tokens, labels)
        else:
            inputs_embeds = self.model.embed_tokens(input_ids)

        outputs = self.model(
            input_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.llm_head(hidden_states)

        loss = None
        # we're training! set up loss functions
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def _combine_image_text_embeds(
        self,
        input_ids: torch.LongTensor,
        img_tokens: torch.FloatTensor
    ):
        # basically, look for image start tokens in the input token ids,
        # insert the image embeddings where the image start token is within the text embeddings
        # while also inserting IGNORE_INDEX label into the modified label tensor at image start token
        new_input_embeds = []
        new_labels = []
        image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # lets find all the chunks seperated by the image token index
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

            if len(image_token_indices) > 0:
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                image_idx += 1
                continue

            cur_input_embeds = []
            batch_img_idx = 0
            image_token_start = image_token_indices[batch_img_idx]
            while batch_img_idx < image_token_indices:
                cur_img_tokens = img_tokens[image_idx]
                image_token_start = image_token_indices[batch_img_idx]
                cur_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                cur_input_embeds.append(cur_img_tokens)

                image_idx += 1
                batch_img_idx += 1

            if image_token_start + 1 < len(cur_input_ids):
                cur_input_embeds.append(cur_input_ids[image_token_start + 1:])

            cur_input_embeds = [x.to(device=self.device) for x in cur_input_embeds]
            cur_input_embeds = torch.cat(cur_input_embeds, dim=0)
            new_input_embeds.append(cur_input_embeds)

        return new_input_embeds

    def _combine_image_text_labels(
        self,
        input_ids: torch.LongTensor,
        labels: torch.FloatTensor,
        img_tokens: torch.FloatTensor
    ):
        new_labels = []
        image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # lets find all the chunks seperated by the image token index
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

            if len(image_token_indices) > 0:
                new_labels.append(labels[batch_idx])
                image_idx += 1
                continue

            cur_new_labels = []
            cur_labels = labels[batch_idx]
            assert cur_labels.shape == cur_input_ids.shape

            batch_img_idx = 0
            image_token_start = image_token_indices[batch_img_idx]
            while batch_img_idx < image_token_indices:
                cur_img_tokens = img_tokens[image_idx]
                image_token_start = image_token_indices[batch_img_idx]
                cur_new_labels.append(cur_labels[:image_token_start])
                cur_new_labels.append(torch.full((cur_img_tokens.shape[0]), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                cur_labels = cur_labels[image_token_start+1:]

                image_idx += 1
                batch_img_idx += 1

            if image_token_start + 1 < len(cur_input_ids):
                cur_new_labels.append(cur_labels)

            cur_new_labels = torch.cat(cur_new_labels, dim=0)
            new_labels.append(cur_new_labels)

        return new_labels

    def _generate_img_tokens(self, images: torch.FloatTensor):
        with torch.no_grad():
            proc_images = self.image_processor(images)
            img_feat = self.vis_enc(proc_images)
            Z = img_feat.hidden_states[self.config.img_hidden_states_layer]
        H = self.image_projector(Z)
        return H


