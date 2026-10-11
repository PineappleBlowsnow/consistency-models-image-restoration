# Image Restoration with Consistency Models and Diffusion

**MVA course — Generative Models for Images**

A collaborative project by **Ying Jin and Ianis Hammani** studying consistency training/distillation and using pretrained generative models for image restoration.

The work starts with a controlled two-moons example, then compares consistency-model and diffusion-posterior-sampling approaches to inpainting. The image experiments examine restoration quality alongside network evaluations and wall-clock latency.

## What to explore

| Experiment | Question | Source |
|---|---|---|
| Consistency training and distillation | How do the two approaches behave on a simple 2D distribution? | [Two-moons notebook](A_Study_of_Consistency_Models_make_moons.ipynb) |
| Image restoration | How do pretrained consistency models perform on inpainting and colorization? | [Image-restoration notebook](CM_Inpainting_%2B_Compare_with_DPS.ipynb) |
| CM versus DPS | What quality is obtained at different inference budgets? | [Benchmark implementation](dps_benchmark.py) and [DPS tuning notebook](DPS_Finetuning_Only.ipynb) |

The [joint report](report.pdf) contains the original experiment discussion. Authorship is shared; the archived materials do not establish a complete per-person implementation split.

## Models, data and evaluation

- The image notebook uses `openai/diffusers-cd_bedroom256_lpips`, distilled on LSUN Bedroom at 256×256, and the LSUN Bedroom diffusion checkpoint `lsun_bedroom.pt` for DPS.
- Reference images come from the LSUN Bedroom validation set. Checkpoints and image data must be obtained separately.
- The experiments consider PSNR, SSIM and LPIPS, together with network function evaluations (NFE) and elapsed inference time.
- The audited inpainting comparison uses zero measurement noise. The broader noisy-inverse-problem motivation should not be read as proof of a validated noisy benchmark.

## Reading and running

Start with the report and notebook sections, then inspect the relevant setup cells before running. The notebooks retain their original Colab/Drive setup and external checkpoint paths; adapt these to your own environment. This snapshot does not yet provide a tested environment lock or a single-command reproduction.

The original image notebook and report are large (approximately 90 MB and 70 MB). They are retained with their historical outputs; [the original result folder](https://drive.google.com/drive/folders/1mJg4ZqW1pEvy2PW_9MbU4vLllNnhiVGq?usp=sharing) is an additional reference, not a replacement for the source code.

## References and status

- [Consistency Models](https://arxiv.org/abs/2303.01469)
- The pretrained consistency and diffusion models, and external DPS implementation, retain their original attribution.

This is a course implementation and experimental study. The portfolio documentation was reorganized on 11 October 2026; training and image benchmarks were not rerun in this update.


[Back to Ying Jin's project directory](https://github.com/PineappleBlowsnow#mva-projects)
