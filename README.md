# Consistency Models Project

This repository contains two main notebooks about project 8 on Consistency Models (CMs).

The main work is:

1. make_moons/make_moons/A_Study_of_Consistency_Models_make_moons.ipynb
2. zero shot photo edit/zero shot photo edit/CM_Inpainting_+_Compare_with_DPS.ipynb

dps_benchmark.py is the supporting document for second notebook. DPS_Finetuning_Only.ipynb is the file to find good hypermeters for DPS. 

## Notebook 1

### `A_Study_of_Consistency_Models_make_moons.ipynb`

This notebook studies Consistency Models on the two-moons dataset, which is a simple 2D toy dataset.

The notebook:

- creates and normalizes the two-moons data
- trains a diffusion teacher model
- trains a Consistency Model with two methods:
  - **Consistency Distillation (CD)**
  - **Consistency Training (CT)**
- compares generated samples and training behavior
- evaluates generation quality with plots and metrics

This notebook is mainly for understanding how Consistency Models work in a small and controlled setting.

## Notebook 2

### `CM_Inpainting_+_Compare_with_DPS.ipynb`

This notebook is mainly about applying a pretrained Consistency Model to image restoration problems without retraining the model. It contains two parts of work: 

- Evalutate CM's zero shot photo edit performance on two tasks: inpainting and colorization using metrics including PSNR, SSIM, LPIPS. 
- Compare CM with DPS on inpainting by benchmarking perceptual fidelity (LPIPS) against both computational budget (NFE) and empirical inference latency (wall-clock time).


For the experiments, we uses the OpenAI pretrained checkpoint `openai/diffusers-cd_bedroom256_lpips`, which is a Consistency Model distilled on **LSUN Bedroom 256x256**. For DPS, we use the **LSUN Bedroom diffusion prior** checkpoint `lsun_bedroom.pt` 
In the notebook, this pretrained CM is used as the image prior for **zero-shot image editing**. The notebook loads reference images from the **LSUN Bedroom validation set** and resizes them to **256x256**, which matches the pretrained checkpoint.





## Outputs

All outputs are saved in the google drive: https://drive.google.com/drive/folders/1mJg4ZqW1pEvy2PW_9MbU4vLllNnhiVGq?usp=sharing 
