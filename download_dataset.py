import os
import modal
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

VOLUME_DIR="/dataset"

web_app = FastAPI()
stub = modal.Stub("llava_dataset_downloader")
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
    .run_commands("python -m pip install flash-attn --no-build-isolation", gpu="any")
    .copy_local_file(local_path="download_dataset.py", remote_path="/download_dataset.py")
)

@stub.function(
    image=image,
    timeout=82800,  # 30 minutes
    network_file_systems={VOLUME_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface")]
)
def download_dataset():
    import datasets
    dataset = datasets.load_dataset("liuhaotian/LLaVA-Pretrain", data_files=["blip_laion_cc_sbu_558k.json"], cache_dir=VOLUME_DIR)
    images = datasets.load_dataset("liuhaotian/LLaVA-Pretrain", data_files=["images.zip"], cache_dir=VOLUME_DIR)
    del dataset
    del images


@stub.local_entrypoint()
def run():
    download_dataset.remote()
