import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import datasets
import transformers
from models.llava_zephyr import LlavaZephyrModel
from models.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def tokenize_convos(examples, tokenizer):
    system="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
    wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
    sep=" "
    sep2="</s>"

    prompts = []
    for convo in examples["conversations"]:
        ret = ""
        for i, message_dict in enumerate(convo):
            message = message_dict["value"]
            if i == 0:
                message = wrap_sys(system) + message
                message = wrap_inst(message)
                ret += sep + message
            else:
                ret += " " + message + " " + sep2
        
        ret = ret.lstrip(sep)
        prompts.append(ret)

    tokenized_batch = {}
    if "image" in examples:
        tokenized_tensors = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in prompts]
        tokenized_batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
            tokenized_tensors,
            batch_first=True,
            padding_value=tokenizer.pad_token_id)
        tokenized_batch["attention_masks"] = tokenized_batch["input_ids"].ne(tokenizer.pad_token_id)
    else:
        tokenized_batch = tokenizer(prompts, return_tensors="pt", padding="longest")
    tokenized_batch["prompt"] = prompts
    return tokenized_batch

def get_labels_from_input_ids(conversations, input_ids, has_image):
    targets = input_ids.clone()
    sep = "[/INST] "
    sep2="</s>",
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    
train_cache_dir = "output/"
model_path = "HuggingFaceH4/zephyr-7b-beta"
model_max_length = 8000
ccm3_dataset_json = "/home/yan/.cache/huggingface/datasets/downloads/9d59a1fc0001fcd85430e5e31e232abeb12246176629adec535eb3aa8f960a8b"
image_folder = "/home/yan/.cache/huggingface/datasets/downloads/extracted/8c24317d562478024415867912e51ffc893145f0c92d28aeb37621cac36e4760"
dataset = datasets.load_dataset('json', data_files=ccm3_dataset_json)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_path,
    cache_dir=train_cache_dir,
    model_max_length=model_max_length,
    padding_side="right",
    use_fast=False,
)
tokenizer.pad_token = tokenizer.unk_token

proc_convo = dataset["train"].map(
    lambda examples: tokenize_convos(examples, tokenizer),
    batched=True, batch_size=16)
print(proc_convo[0])

model = LlavaZephyrModel.from_pretrained(model_path, cache_dir=train_cache_dir)
model.config.use_cache = False
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):  # number of epochs
    model.train()
    for batch in proc_convo:
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = get_labels_from_input_ids(batch["prompt"])

        if "image" in batch:
          images = [torch.FloatTensor(os.path.join(image_folder, image_path)) for image_path in batch["image"]]
        else:
          images = None

        outputs = model(inputs, attention_mask=attention_mask, labels=labels, images=images)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} completed. Loss - {loss.item()}")
