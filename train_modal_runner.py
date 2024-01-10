import os
import modal
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

VOLUME_DIR="/dataset"
web_app = FastAPI()
stub = modal.Stub("llava_finetune_app")
volume = modal.NetworkFileSystem.from_name("llava-dataset")

mount = modal.Mount.from_local_dir(".", remote_path="/")
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel",
        setup_dockerfile_commands=[
            "RUN ln -snf /usr/share/zoneinfo/Europe/Minsk /etc/localtime && echo Europe/Minsk > /etc/timezone",
            "RUN apt-get update && apt-get install -y git python3-pip gcc build-essential libgl1 libglib2.0-0 ffmpeg wget freeglut3-dev",
            "RUN ln -s /usr/bin/python3 /usr/bin/python",
        ],
    )
    .pip_install_from_requirements("requirements.txt", gpu="any")
    .run_commands("find / | grep nvcc", gpu="any")
    .run_commands("python -m pip install flash-attn --no-build-isolation", gpu="any")
    .copy_local_dir(local_path="models", remote_path="/models")
    .copy_local_file(local_path="train_baby_volcano.py", remote_path="/train_baby_volcano.py")
    .copy_local_file(local_path="train_modal_runner.py", remote_path="/train_modal_runner.py")
)

@stub.function(
    image=image,
    network_file_systems={VOLUME_DIR: volume},
    gpu=modal.gpu.A100(memory=80),  # finetuning is VRAM hungry, so this should be an A100
    timeout=86400,  # 30 minutes
    secrets=[modal.Secret.from_name("huggingface")]
)
def train():
    import sys
    sys.path.append("/")
    import train_baby_volcano
    train_baby_volcano.train()


@stub.local_entrypoint()
def run():
    train.remote()
