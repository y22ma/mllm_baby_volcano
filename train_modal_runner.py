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
    gpu=modal.gpu.A100(memory=80),  # finetuning is VRAM hungry, so this should be an A100
    timeout=86400,  # 30 minutes
    secrets=[modal.Secret.from_name("huggingface")]
)
def train():
    import sys
    sys.path.append("/")
    import train_llava_zephyr
    train_llava_zephyr.train()


@stub.local_entrypoint()
def run():
    train.remote()
