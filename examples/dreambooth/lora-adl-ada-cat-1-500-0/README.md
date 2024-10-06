---
license: openrail++
library_name: diffusers
tags:
- text-to-image
- text-to-image
- diffusers-training
- diffusers
- lora
- template:sd-lora
- stable-diffusion-xl
- stable-diffusion-xl-diffusers
base_model: SDXL_model
instance_prompt: a photo of adl cat
widget:
- text: a photo of adl cat by the sea
  output:
    url: image_0.png
- text: a photo of adl cat by the sea
  output:
    url: image_1.png
- text: a photo of adl cat by the sea
  output:
    url: image_2.png
- text: a photo of adl cat by the sea
  output:
    url: image_3.png
---

<!-- This model card has been generated automatically according to the information the training script had access to. You
should probably proofread and complete it, then remove this comment. -->


# SDXL LoRA DreamBooth - YongjieNiu/lora-adl-ada-cat-1-500-0

<Gallery />

## Model description

These are YongjieNiu/lora-adl-ada-cat-1-500-0 LoRA adaption weights for SDXL_model.

The weights were trained  using [DreamBooth](https://dreambooth.github.io/).

LoRA for the text encoder was enabled: False.

Special VAE used for training: VAE.

## Trigger words

You should use a photo of adl cat to trigger the image generation.

## Download model

Weights for this model are available in Safetensors format.

[Download](YongjieNiu/lora-adl-ada-cat-1-500-0/tree/main) them in the Files & versions tab.



## Intended uses & limitations

#### How to use

```python
# TODO: add an example code snippet for running this diffusion pipeline
```

#### Limitations and bias

[TODO: provide examples of latent issues and potential remediations]

## Training details

[TODO: describe the data used to train the model]