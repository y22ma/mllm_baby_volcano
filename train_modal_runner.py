import os
import modal
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

def download_models():
    from transformers import AutoModel
    model = AutoModel.from_pretrained("HuggingFaceH4/zephyr-7b-beta", cache_dir="/dataset/")
    model = AutoModel.from_pretrained("facebook/dinov2-large", cache_dir="/dataset/")

def download_datasets():
    import datasets
    dataset = datasets.load_dataset("liuhaotian/LLaVA-Pretrain", data_files=["blip_laion_cc_sbu_558k.json"], cache_dir="/dataset/")

def download_images():
    import datasets
    images = datasets.load_dataset("liuhaotian/LLaVA-Pretrain", data_files=["images.zip"], cache_dir="/dataset/")


web_app = FastAPI()
stub = modal.Stub("llava_finetune_app")

cache_dir = "/llava-dabble-model"
volume = modal.NetworkFileSystem.persisted("llava-finetune-data")
mount = modal.Mount.from_local_dir(".", remote_path="/")
image = (
    modal.Image.debian_slim()
    .apt_install("fonts-freefont-ttf", "git", "libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install_from_requirements("requirements.txt", gpu="any")
    .run_function(download_models)
    #.run_function(download_datasets)
    #.run_function(download_images)
    .copy_local_dir(local_path="models", remote_path="/models")
    .copy_local_file(local_path="train_llava_zephyr.py", remote_path="/train_llava_zephyr.py")
    .copy_local_file(local_path="train_modal_runner.py", remote_path="/train_modal_runner.py")
)


@stub.function(
    image=image,
    gpu="A100",  # finetuning is VRAM hungry, so this should be an A100
    timeout=86400,  # 30 minutes
    network_file_systems={cache_dir:volume},
    secrets=[modal.Secret.from_name("huggingface")]
)
def train():
    import huggingface_hub
    import os
    import sys
    import copy
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.transforms as transforms 
    import numpy as np
    import datasets
    import transformers
    
    from typing import Dict, Optional, Sequence, List
    from PIL import Image

    sys.path.append("/")
    from models.llava_zephyr import LlavaZephyrModelForCausalLM
    from models.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    
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
    
    def preprocess_multimodal(
        sources: Sequence[str],
    ) -> Dict:
        for source in sources:
            for sentence in source:
                if DEFAULT_IMAGE_TOKEN in sentence['value']:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                    sentence['value'] = sentence['value'].strip()
                replace_token = DEFAULT_IMAGE_TOKEN
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    
        return sources
    
    def tokenize_convos(examples, tokenizer):
        system="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
        wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
        sep=" "
        sep2="</s>"
    
        prompts = []
        preprocess_multimodal(copy.deepcopy(examples["conversations"]))
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
            tokenized_batch["input_ids"] = [tensor for tensor in tokenized_batch["input_ids"]]
            tokenized_batch["attention_masks"] = [
                tensor.ne(tokenizer.pad_token_id) for tensor in tokenized_batch["input_ids"]]
        else:
            tokenized_batch = tokenizer(prompts, return_tensors="pt", padding="longest")
        tokenized_batch["prompt"] = prompts
        return tokenized_batch
    
    def get_labels_from_input_ids(conversations, input_ids, has_image):
        targets = input_ids.clone()
        sep = "[/INST] "
        sep2="</s>"
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

        return targets
    
    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
    
    print("Logging in huggingface")
    hf_key = os.environ["HUGGINGFACE_TOKEN"]
    huggingface_hub.login(hf_key)

    print("Downloading models!")
    train_cache_dir = "cache/"
    model_path = "HuggingFaceH4/zephyr-7b-beta"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=train_cache_dir,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device {}".format(device))
    model = LlavaZephyrModelForCausalLM.from_pretrained(model_path, cache_dir=train_cache_dir)
    model.config.use_cache = False
    model = model.to(device)
    processor = transformers.AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    
    # loading dataset
    print("Downloading dataset json!")
    dataset_path = "liuhaotian/LLaVA-Pretrain"
    dataset_path = "y22ma/LLAVA-Test"
    #dataset = datasets.load_dataset(dataset_path, data_files=["blip_laion_cc_sbu_558k.json"], cache_dir="dataset/")
    dataset = datasets.load_dataset(dataset_path, data_files=["test.json"], cache_dir="/dataset/")
    proc_convo = dataset["train"].map(
        lambda examples: tokenize_convos(examples, tokenizer),
        batched=True, batch_size=16)
    
    print("Downloading images!")
    images = datasets.load_dataset(dataset_path, data_files=["images.zip"], cache_dir="/dataset/")
    images = images.cast_column("image", datasets.Image(decode=False))
    path_divs = images["train"]["image"][0]["path"].split('/')
    image_folder = os.path.join("/", *path_divs[:-2])
    
    # training loop!
    print("Starting Training!")
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
    for epoch in range(10):  # number of epochs
        for batch in proc_convo.iter(batch_size=16):
            has_image = "image" in batch
            input_ids = torch.Tensor(batch['input_ids']).long().to(device)
            attention_mask = torch.Tensor(batch['attention_masks']).to(device)
            print("hi")
    
            labels = get_labels_from_input_ids(batch["prompt"], input_ids, has_image).to(device)
            print("hi2")
            images = None
            if "image" in batch:
                images = []
                for img_path in batch["image"]:
                    full_path = os.path.join(image_folder, img_path)
                    image = Image.open(full_path).convert('RGB')
                    image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    images.append(image)
                images = torch.stack(images).float().to(device)
            print("hi3")
    
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, images=None)
            loss = outputs.loss
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1} Loss - {loss.item()}")



@stub.local_entrypoint()
def run():
    train.remote()
