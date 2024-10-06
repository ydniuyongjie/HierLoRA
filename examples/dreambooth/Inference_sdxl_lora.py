import argparse
import ast
import json
import os
import shutil

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file

from diffusers import AutoencoderKL, StableDiffusionXLPipeline


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a inference  script.")
    parser.add_argument(
        "--lora_model_id",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained lora or lora identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--new_prompt",
        type=str,
        default=None,
        help=("The prompt or prompts to guide the image generation."),
    )
    parser.add_argument(
        "--old_prompt",
        type=str,
        default=None,
        help=("The prompt or prompts to guide the image generation."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=("The directory where to save the exported images."),
    )
    parser.add_argument(
        "--num_images",
        default=4,
        help="The number of images",
        type=int,
    )
    args = parser.parse_args(input_args)
    return args


def convert_to_list(s):
    # 移除花括号，分割字符串，并去除引号
    return [item.strip("'") for item in s.strip("{}").split(", ")]


def remap_lora_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'lora' in key:
            # 添加前缀并在 weight 前插入 default
            new_key = f"base_model.model.{key.rsplit('.weight', 1)[0]}.default.weight"
            new_state_dict[new_key] = value
        else:
            # 对于非 LoRA 权重，保持原样
            new_state_dict[key] = value
    return new_state_dict
def load_lora_model(pipe, load_path):
    # 加载 LoRA 配置
    with open(f"{load_path}/lora_config.json", "r") as f:
        config_dict = json.load(f)
    # 转换 target_modules
    target_modules = convert_to_list(config_dict.get("target_modules"))
    # 创建 LoraConfig，确保包含 RSLoRA 参数
    lora_config = LoraConfig(
        r=config_dict["r"],
        lora_alpha=config_dict["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=config_dict["lora_dropout"],
        bias=config_dict["bias"],
        use_rslora=config_dict["use_rslora"],
        use_active_func=config_dict["use_active_func"],
    )
    # 应用 LoRA 配置到 U-Net
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    # 加载 U-Net LoRA 权重
    unet_lora_layers = load_file(f"{load_path}/unet_lora_weights.safetensors")

    # for key in pipe.unet.state_dict().keys():
    #     print(key)
    # # print("Model keys:", pipe.unet.state_dict().keys())
    # # print("Loaded keys:", unet_lora_layers.keys())
    # for key in unet_lora_layers.keys():
    #     print(key,unet_lora_layers[key])
    new_unet_lora_layers = remap_lora_keys(unet_lora_layers)
    # for key in new_unet_lora_layers.keys():
    #     print(key,new_unet_lora_layers[key])
    pipe.unet.load_state_dict(new_unet_lora_layers, strict=False)
    print(f"LoRA model loaded from {load_path}")

    return pipe

if __name__ == "__main__":
    args = parse_args()
    lora_model_id = args.lora_model_id
    base_model = "SDXL_model"
    VAE_model = "VAE"

    pipe = StableDiffusionXLPipeline.from_pretrained(base_model, torch_dtype=torch.float32)
    vae = AutoencoderKL.from_pretrained(VAE_model, torch_dtype=torch.float32)
    pipe.vae = vae
    #加载模型的配置文件和权重
    pipe = load_lora_model(pipe, lora_model_id)
    # pipe.load_lora_weights(lora_model_id, weight_name="pytorch_lora_weights.safetensors")

    pipe.to("cuda")

    # new prompt
    prompt = args.new_prompt
    output_dir = args.output_dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir+'/new/')
    images = [pipe(prompt, num_inference_steps=25).images[0] for i in range(args.num_images)]
    for i, image in enumerate(images):
        image.save(f"{output_dir+'/new/'}/cat{i}.png")
    # old prompt
    prompt = args.old_prompt
    output_dir = args.output_dir
    os.makedirs(output_dir+'/old/')
    images = [pipe(prompt, num_inference_steps=25).images[0] for i in range(args.num_images)]
    for i, image in enumerate(images):
        image.save(f"{output_dir+'/old/'}/cat{i}.png")

