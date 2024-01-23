from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn

from .constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from transformers import AutoModel, MistralModel, MistralConfig, MistralForCausalLM, CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class LlavaZephyrModelForCausalLM(MistralForCausalLM):
    def __init__(self, config: MistralConfig):
        super(MistralForCausalLM, self).__init__(config)
        self.config = config
        print(self.config)
        self.model = MistralModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def set_vis_enc(self):
        self.vis_enc = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
        self.image_projector = nn.Linear(self.vis_enc.config.hidden_size, self.config.hidden_size)

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
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        if images is not None:
            # pass the image tensor throught the image encoder and the projector to get the
            # image tokens, referred to as H in the original LLAVA paper
            img_tokens = self._generate_img_tokens(images)
            inputs_embeds = self._combine_image_text_embeds(input_ids, img_tokens)
            new_labels, new_labels_list = self._combine_image_text_labels(
                input_ids, labels, img_tokens)
            attention_mask = self._combine_image_text_attn_masks(
                labels, new_labels, new_labels_list, attention_mask)
            labels = new_labels
        else:
            inputs_embeds = self.model.embed_tokens(input_ids)

        print("forwarding model")
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        print("llm execution finished")

        hidden_states = outputs.last_hidden_state.to(dtype=torch.float16)
        logits = self.lm_head(hidden_states)

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

        print("loss constructed")

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

    def get_model(self):
        return self.model

    def _combine_image_text_embeds(
        self,
        input_ids: torch.LongTensor,
        img_tokens: torch.FloatTensor
    ):
        # basically, look for image start tokens in the input token ids,
        # insert the image embeddings where the image start token is within the text embeddings
        # while also inserting IGNORE_INDEX label into the modified label tensor at image start token
        new_input_embeds = []
        image_idx = 0
        for cur_input_ids in input_ids:
            # lets find all the chunks seperated by the image token index
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

            if len(image_token_indices) == 0:
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                image_idx += 1
                continue

            cur_input_embeds = []
            batch_img_idx = 0
            image_token_start = 0
            while batch_img_idx < len(image_token_indices):
                cur_img_tokens = img_tokens[image_idx]
                image_token_end = image_token_indices[batch_img_idx]
                cur_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start:image_token_end]))
                cur_input_embeds.append(cur_img_tokens)

                image_token_start = image_token_end + 1
                image_idx += 1
                batch_img_idx += 1

            if image_token_end + 1 < len(cur_input_ids):
                cur_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_end + 1:]))

            cur_input_embeds = [x.to(device=self.device) for x in cur_input_embeds]
            cur_input_embeds = torch.cat(cur_input_embeds, dim=0)
            new_input_embeds.append(cur_input_embeds)

        max_len = max(x.shape[0] for x in new_input_embeds)

        new_input_embeds_align = []
        for cur_new_embed in new_input_embeds:
            cur_new_embed = torch.cat(
                (cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
            new_input_embeds_align.append(cur_new_embed)
        new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

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

            if len(image_token_indices) == 0:
                new_labels.append(labels[batch_idx])
                image_idx += 1
                continue

            cur_new_labels = []
            cur_labels = labels[batch_idx]
            assert cur_labels.shape == cur_input_ids.shape

            batch_img_idx = 0
            image_token_start = 0
            while batch_img_idx < len(image_token_indices):
                cur_img_tokens = img_tokens[image_idx]
                image_token_end = image_token_indices[batch_img_idx]
                cur_new_labels.append(cur_labels[image_token_start:image_token_end])
                cur_new_labels.append(torch.full((cur_img_tokens.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))

                image_idx += 1
                batch_img_idx += 1

            if image_token_end + 1 < len(cur_input_ids):
                cur_new_labels.append(cur_labels[image_token_end + 1:])

            cur_new_labels = torch.cat(cur_new_labels, dim=0)
            new_labels.append(cur_new_labels)

        max_len = max(x.shape[0] for x in new_labels)
        new_labels_align = []
        new_labels_list = new_labels
        for cur_new_label in new_labels:
            cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
            new_labels_align.append(cur_new_label)
        new_labels = torch.stack(new_labels_align, dim=0)

        return new_labels, new_labels_list


    def _combine_image_text_attn_masks(
        self, labels, new_labels, new_labels_list, attention_mask
    ):
        new_attention_mask = []
        for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, new_labels_list, new_labels):
            new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
            new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
            cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
            new_attention_mask.append(cur_new_attention_mask)
        attention_mask = torch.stack(new_attention_mask, dim=0)

        return attention_mask

    def _generate_img_tokens(self, images: torch.FloatTensor):
        with torch.no_grad():
            img_feat = self.vis_enc(images.to(device=self.vis_enc.device, dtype=self.vis_enc.dtype), output_hidden_states=True)
            Z = img_feat.hidden_states[-1]
            Z = Z[:, 1:]
        H = self.image_projector(Z)
        return H


