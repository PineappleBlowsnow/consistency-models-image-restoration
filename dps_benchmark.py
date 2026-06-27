from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


LSUN_BEDROOM_DIFFUSION_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt"
DPS_REPO_URL = "https://github.com/DPS2022/diffusion-posterior-sampling.git"
MOTIONBLUR_REPO_URL = "https://github.com/LeviBorodenko/motionblur.git"


def ensure_git_repo(repo_dir: Path, repo_url: str) -> Path:
    if repo_dir.exists() and any(repo_dir.iterdir()):
        return repo_dir
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
    return repo_dir


def build_random_dropout_mask(size: int, missing_fraction: float = 0.80, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand((1, 1, size, size), generator=generator) < missing_fraction).float()


def seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class DPSBedroomRunner:
    def __init__(
        self,
        base_dir: Path,
        device: str,
        in_colab: bool,
        use_fp16: bool,
        seed: int = 0,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.device = device
        self.in_colab = in_colab
        self.use_fp16 = use_fp16
        self.seed = seed

        self.dps_repo_dir = Path("/content/diffusion-posterior-sampling" if in_colab else self.base_dir / "diffusion-posterior-sampling")
        self.motionblur_dir = Path("/content/motionblur" if in_colab else self.base_dir / "motionblur")
        self.model_dir = self.dps_repo_dir / "models"
        self.checkpoint_path = self.model_dir / "lsun_bedroom.pt"

        self.model_config = {
            "image_size": 256,
            "num_channels": 256,
            "num_res_blocks": 2,
            "channel_mult": "",
            "learn_sigma": True,
            "class_cond": False,
            "use_checkpoint": False,
            "attention_resolutions": "32,16,8",
            "num_heads": 4,
            "num_head_channels": 64,
            "num_heads_upsample": -1,
            "use_scale_shift_norm": True,
            "dropout": 0.1,
            "resblock_updown": True,
            "use_fp16": bool(use_fp16 and device == "cuda"),
            "use_new_attention_order": False,
        }
        self.diffusion_config = {
            "sampler": "ddpm",
            "steps": 1000,
            "noise_schedule": "linear",
            "model_mean_type": "epsilon",
            "model_var_type": "learned_range",
            "dynamic_threshold": False,
            "clip_denoised": True,
            "rescale_timesteps": False,
        }

        self.modules = None
        self.model = None
        self.sampler_cache: dict[int, object] = {}

    def ensure_environment(self) -> dict:
        ensure_git_repo(self.dps_repo_dir, DPS_REPO_URL)
        ensure_git_repo(self.motionblur_dir, MOTIONBLUR_REPO_URL)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        if not self.checkpoint_path.exists():
            print(f"Downloading LSUN Bedroom diffusion prior from {LSUN_BEDROOM_DIFFUSION_URL}")
            urlretrieve(LSUN_BEDROOM_DIFFUSION_URL, self.checkpoint_path)

        repo_paths = [str(self.dps_repo_dir.resolve()), str(self.motionblur_dir.parent.resolve())]
        for repo_path in reversed(repo_paths):
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)

        for module_name in list(sys.modules):
            if module_name == "guided_diffusion" or module_name.startswith("guided_diffusion."):
                sys.modules.pop(module_name, None)
            if module_name == "util" or module_name.startswith("util."):
                sys.modules.pop(module_name, None)
            if module_name == "motionblur" or module_name.startswith("motionblur."):
                sys.modules.pop(module_name, None)

        from guided_diffusion.condition_methods import get_conditioning_method
        from guided_diffusion.measurements import get_noise, get_operator
        from guided_diffusion.unet import create_model
        from guided_diffusion.gaussian_diffusion import create_sampler

        self.modules = {
            "get_conditioning_method": get_conditioning_method,
            "get_noise": get_noise,
            "get_operator": get_operator,
            "create_model": create_model,
            "create_sampler": create_sampler,
        }
        return self.modules

    def load_prior(self, force_reload: bool = False):
        if self.model is not None and not force_reload:
            return self.model
        if self.modules is None:
            self.ensure_environment()
        create_model = self.modules["create_model"]
        model = create_model(**self.model_config, model_path=str(self.checkpoint_path))
        if self.model_config["use_fp16"] and self.device == "cuda":
            model.convert_to_fp16()
        model = model.to(self.device)
        model.eval()
        self.model = model
        return self.model

    def get_sampler(self, num_inference_steps: int):
        num_inference_steps = int(num_inference_steps)
        if num_inference_steps not in self.sampler_cache:
            if self.modules is None:
                self.ensure_environment()
            create_sampler = self.modules["create_sampler"]
            self.sampler_cache[num_inference_steps] = create_sampler(
                **self.diffusion_config,
                timestep_respacing=num_inference_steps,
            )
        return self.sampler_cache[num_inference_steps]

    @staticmethod
    def make_known_mask(missing_mask: torch.Tensor, channels: int = 3) -> torch.Tensor:
        if missing_mask.shape[1] == channels:
            return 1.0 - missing_mask
        return (1.0 - missing_mask).repeat(1, channels, 1, 1)

    def run(
        self,
        image: torch.Tensor,
        missing_mask: torch.Tensor,
        num_inference_steps: int,
        noise_sigma: float,
        conditioning_scale: float,
        seed: int,
        make_noisy_observation,
        record_history: bool = False,
    ):
        if self.modules is None:
            self.ensure_environment()
        model = self.load_prior()
        model_dtype = next(model.parameters()).dtype

        image = image.to(device=self.device, dtype=model_dtype)
        missing_mask = missing_mask.to(device=self.device, dtype=model_dtype)
        known_mask = self.make_known_mask(missing_mask, channels=image.shape[1]).to(device=self.device, dtype=model_dtype)
        measurement = make_noisy_observation(image, missing_mask[:, :1], noise_sigma=noise_sigma, seed=seed).to(
            device=self.device,
            dtype=model_dtype,
        )

        sampler = self.get_sampler(num_inference_steps)
        operator = self.modules["get_operator"](name="inpainting", device=self.device)
        noiser = self.modules["get_noise"](name="gaussian", sigma=float(noise_sigma))
        cond_method = self.modules["get_conditioning_method"]("ps", operator, noiser, scale=conditioning_scale)

        seed_torch(seed)
        img = torch.randn_like(image, device=self.device, dtype=model_dtype)
        history = [img.detach().float().cpu()] if record_history else []

        synchronize_device()
        start_time = time.perf_counter()
        for idx in reversed(range(sampler.num_timesteps)):
            t = torch.full((img.shape[0],), idx, device=self.device, dtype=torch.long)
            img = img.requires_grad_()
            out = sampler.p_sample(model=model, x=img, t=t)
            img, _ = cond_method.conditioning(
                x_prev=img,
                x_t=out["sample"],
                x_0_hat=out["pred_xstart"],
                measurement=measurement,
                mask=known_mask,
            )
            img = img.detach().clamp(-1.0, 1.0)
            if record_history:
                history.append(img.detach().float().cpu())

        synchronize_device()
        wall_clock_s = time.perf_counter() - start_time
        return img.detach().float().cpu(), history, wall_clock_s


def build_case_id(reference_index: int, mask_name: str, noise_sigma: float) -> str:
    return f"ref_{reference_index:05d}_{mask_name}_sigma_{noise_sigma:.2f}"


def run_cm_inpainting_benchmark(
    reference_entries,
    comparison_masks,
    noise_levels,
    steps,
    seed,
    pipe,
    zero_shot_image_editing,
    identity_operator,
    make_noisy_observation,
    compute_metrics,
    apply_mask,
):
    cm_results = {}
    metric_rows = []
    mask_names = list(comparison_masks.keys())

    for ref_entry in reference_entries:
        reference = ref_entry["tensor"]
        for mask_name in mask_names:
            missing_mask = comparison_masks[mask_name].clone()
            for noise_sigma in noise_levels:
                observation = make_noisy_observation(reference, missing_mask, noise_sigma=noise_sigma, seed=seed)
                case_id = build_case_id(ref_entry["index"], mask_name, noise_sigma)
                cm_results[case_id] = {
                    "method": "CM",
                    "reference_index": ref_entry["index"],
                    "reference_source": ref_entry["source"],
                    "reference_tensor": reference,
                    "mask_name": mask_name,
                    "mask": missing_mask,
                    "masked": observation,
                    "masked_display": apply_mask(observation, missing_mask, fill_value=-1.0),
                    "noise_sigma": noise_sigma,
                    "runs": {},
                }

                for step in steps:
                    synchronize_device()
                    start_time = time.perf_counter()
                    edited, history = zero_shot_image_editing(
                        pipe,
                        observation,
                        missing_mask,
                        A=identity_operator,
                        A_inv=identity_operator,
                        num_inference_steps=step,
                        seed=seed,
                    )
                    synchronize_device()
                    wall_clock_s = time.perf_counter() - start_time

                    metrics = compute_metrics(reference, observation, edited, missing_mask)
                    metrics["method"] = "CM"
                    metrics["reference_index"] = ref_entry["index"]
                    metrics["reference_source"] = ref_entry["source"]
                    metrics["mask_name"] = mask_name
                    metrics["noise_sigma"] = noise_sigma
                    metrics["steps"] = step
                    metrics["wall_clock_s"] = wall_clock_s
                    metric_rows.append(metrics)

                    cm_results[case_id]["runs"][step] = {
                        "image": edited,
                        "history": history,
                        "wall_clock_s": wall_clock_s,
                    }

    cm_metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["reference_index", "mask_name", "noise_sigma", "steps"]
    ).reset_index(drop=True)
    return cm_results, cm_metrics_df


def run_dps_inpainting_benchmark(
    runner: DPSBedroomRunner,
    reference_entries,
    comparison_masks,
    noise_levels,
    steps,
    seed,
    conditioning_scale,
    make_noisy_observation,
    compute_metrics,
    apply_mask,
):
    dps_results = {}
    metric_rows = []
    mask_names = list(comparison_masks.keys())

    for ref_entry in reference_entries:
        reference = ref_entry["tensor"]
        for mask_name in mask_names:
            missing_mask = comparison_masks[mask_name].clone()
            for noise_sigma in noise_levels:
                observation = make_noisy_observation(reference, missing_mask, noise_sigma=noise_sigma, seed=seed)
                case_id = build_case_id(ref_entry["index"], mask_name, noise_sigma)
                dps_results[case_id] = {
                    "method": "DPS",
                    "reference_index": ref_entry["index"],
                    "reference_source": ref_entry["source"],
                    "reference_tensor": reference,
                    "mask_name": mask_name,
                    "mask": missing_mask,
                    "masked": observation,
                    "masked_display": apply_mask(observation, missing_mask, fill_value=-1.0),
                    "noise_sigma": noise_sigma,
                    "runs": {},
                }

                for step in steps:
                    edited, history, wall_clock_s = runner.run(
                        image=reference,
                        missing_mask=missing_mask,
                        num_inference_steps=step,
                        noise_sigma=noise_sigma,
                        conditioning_scale=conditioning_scale,
                        seed=seed,
                        make_noisy_observation=make_noisy_observation,
                        record_history=False,
                    )

                    metrics = compute_metrics(reference, observation, edited, missing_mask)
                    metrics["method"] = "DPS"
                    metrics["reference_index"] = ref_entry["index"]
                    metrics["reference_source"] = ref_entry["source"]
                    metrics["mask_name"] = mask_name
                    metrics["noise_sigma"] = noise_sigma
                    metrics["steps"] = step
                    metrics["wall_clock_s"] = wall_clock_s
                    metric_rows.append(metrics)

                    dps_results[case_id]["runs"][step] = {
                        "image": edited,
                        "history": history,
                        "wall_clock_s": wall_clock_s,
                    }

    dps_metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["reference_index", "mask_name", "noise_sigma", "steps"]
    ).reset_index(drop=True)
    return dps_results, dps_metrics_df


def save_metrics_table(metrics_df: pd.DataFrame, output_dir: Path, file_name: str, save_outputs: bool = True) -> None:
    if metrics_df.empty:
        print("No metrics to save.")
        return
    if save_outputs:
        output_path = output_dir / file_name
        metrics_df.to_csv(output_path, index=False)
        print(f"Saved metrics to: {output_path}")


def combine_comparison_metrics(cm_metrics: pd.DataFrame | None, dps_metrics: pd.DataFrame | None) -> pd.DataFrame:
    frames = []
    if cm_metrics is not None and not cm_metrics.empty:
        frames.append(cm_metrics.copy())
    if dps_metrics is not None and not dps_metrics.empty:
        frames.append(dps_metrics.copy())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def summarize_comparison_table(comparison_df: pd.DataFrame, noise_sigma: float = 0.0, mask_name: str | None = None) -> pd.DataFrame:
    if comparison_df.empty:
        return pd.DataFrame()
    subset = comparison_df[np.isclose(comparison_df["noise_sigma"], noise_sigma)]
    if mask_name is not None and "mask_name" in subset.columns:
        subset = subset[subset["mask_name"] == mask_name]
    metrics = [
        "missing_region_lpips",
        "missing_region_psnr",
        "missing_region_ssim",
        "known_mae",
        "boundary_mae",
        "inpainted_seam_gap",
        "wall_clock_s",
    ]
    metrics = [metric for metric in metrics if metric in subset.columns]
    if subset.empty or not metrics:
        return pd.DataFrame()
    return subset.groupby(["method", "steps"])[metrics].mean().reset_index()


def show_dps_grid(dps_results: dict, steps, tensor_to_numpy_image, save_outputs: bool, output_dir: Path, figure_name: str):
    if not dps_results:
        print("No DPS results available yet.")
        return
    case_ids = list(dps_results.keys())
    num_rows = len(case_ids)
    num_cols = len(steps) + 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2.8 * num_cols, 2.8 * num_rows), dpi=180)
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    titles = ["Masked input"] + [f"DPS {step}" for step in steps] + ["Reference"]
    for col_idx, title in enumerate(titles):
        axes[0, col_idx].set_title(title, fontsize=11)
    for row_idx, case_id in enumerate(case_ids):
        case = dps_results[case_id]
        axes[row_idx, 0].imshow(tensor_to_numpy_image(case["masked_display"]))
        axes[row_idx, 0].axis("off")
        for col_idx, step in enumerate(steps, start=1):
            if step in case["runs"]:
                axes[row_idx, col_idx].imshow(tensor_to_numpy_image(case["runs"][step]["image"]))
            axes[row_idx, col_idx].axis("off")
        axes[row_idx, -1].imshow(tensor_to_numpy_image(case["reference_tensor"]))
        axes[row_idx, -1].axis("off")
        row_label = f"ref {case['reference_index']} | {case['mask_name']} | sigma={case['noise_sigma']:.2f}"
        axes[row_idx, 0].set_ylabel(row_label, rotation=0, ha="right", va="center", fontsize=10, labelpad=70)
    plt.tight_layout()
    if save_outputs:
        figure_path = output_dir / figure_name
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        print(f"Saved figure to: {figure_path}")
    plt.show()


def show_method_comparison_grid(
    cm_results: dict,
    dps_results: dict,
    reference_index: int,
    mask_name: str,
    noise_sigma: float,
    cm_steps,
    dps_steps,
    tensor_to_numpy_image,
    save_outputs: bool,
    output_dir: Path,
    figure_name: str | None,
):
    case_id = build_case_id(reference_index, mask_name, noise_sigma)
    if case_id not in cm_results or case_id not in dps_results:
        print(f"Case {case_id} is not available in both methods.")
        return

    cm_case = cm_results[case_id]
    dps_case = dps_results[case_id]
    max_steps = max(len(cm_steps), len(dps_steps))
    num_cols = max_steps + 2
    fig, axes = plt.subplots(2, num_cols, figsize=(2.8 * num_cols, 5.6), dpi=180)

    title_steps = cm_steps if len(cm_steps) >= len(dps_steps) else dps_steps
    titles = ["Masked input"] + [f"step {step}" for step in title_steps] + ["Reference"]
    for col_idx, title in enumerate(titles[:num_cols]):
        axes[0, col_idx].set_title(title, fontsize=11)

    axes[0, 0].imshow(tensor_to_numpy_image(cm_case["masked_display"]))
    axes[1, 0].imshow(tensor_to_numpy_image(dps_case["masked_display"]))
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    for col_idx, step in enumerate(cm_steps, start=1):
        axes[0, col_idx].imshow(tensor_to_numpy_image(cm_case["runs"][step]["image"]))
        axes[0, col_idx].axis("off")
    for col_idx in range(len(cm_steps) + 1, num_cols - 1):
        axes[0, col_idx].axis("off")

    for col_idx, step in enumerate(dps_steps, start=1):
        axes[1, col_idx].imshow(tensor_to_numpy_image(dps_case["runs"][step]["image"]))
        axes[1, col_idx].axis("off")
    for col_idx in range(len(dps_steps) + 1, num_cols - 1):
        axes[1, col_idx].axis("off")

    reference_np = tensor_to_numpy_image(cm_case["reference_tensor"])
    axes[0, -1].imshow(reference_np)
    axes[1, -1].imshow(reference_np)
    axes[0, -1].axis("off")
    axes[1, -1].axis("off")

    axes[0, 0].set_ylabel(
        f"CM\nref {reference_index}\n{mask_name}\nsigma={noise_sigma:.2f}",
        rotation=0,
        ha="right",
        va="center",
        fontsize=10,
        labelpad=55,
    )
    axes[1, 0].set_ylabel(
        f"DPS\nref {reference_index}\n{mask_name}\nsigma={noise_sigma:.2f}",
        rotation=0,
        ha="right",
        va="center",
        fontsize=10,
        labelpad=55,
    )

    plt.tight_layout()
    if save_outputs and figure_name is not None:
        figure_path = output_dir / figure_name
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        print(f"Saved comparison figure to: {figure_path}")
    plt.show()


def plot_comparison_curve(
    comparison_df: pd.DataFrame,
    metric_name: str,
    x_axis: str,
    noise_sigma: float,
    mask_name: str | None,
    save_outputs: bool,
    output_dir: Path,
    figure_name: str | None,
):
    if comparison_df.empty:
        print("No comparison data available yet.")
        return
    subset = comparison_df.copy()
    if "noise_sigma" in subset.columns:
        subset = subset[np.isclose(subset["noise_sigma"], noise_sigma)]
    if mask_name is not None and "mask_name" in subset.columns:
        subset = subset[subset["mask_name"] == mask_name]
    if subset.empty or metric_name not in subset.columns or x_axis not in subset.columns:
        print("Requested comparison slice is empty.")
        return

    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=180)
    for method, group in subset.groupby("method"):
        curve = group.groupby(x_axis)[metric_name].mean().reset_index().sort_values(x_axis)
        ax.plot(curve[x_axis], curve[metric_name], marker="o", linewidth=2, label=method)
    ax.set_xlabel("NFE" if x_axis == "steps" else x_axis)
    ax.set_ylabel(metric_name.replace("_", " "))
    title = f"{metric_name} | sigma={noise_sigma:.2f}"
    if mask_name is not None:
        title += f" | {mask_name}"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    if save_outputs and figure_name is not None:
        figure_path = output_dir / figure_name
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        print(f"Saved comparison figure to: {figure_path}")
    plt.show()


def discuss_method_gap(summary_df: pd.DataFrame, title: str) -> None:
    if summary_df.empty:
        print(f"No summary available for {title}.")
        return
    print(title)
    if {"steps", "method", "missing_region_lpips"}.issubset(summary_df.columns):
        lpips_table = summary_df.pivot(index="steps", columns="method", values="missing_region_lpips")
        if {"CM", "DPS"}.issubset(lpips_table.columns):
            best_cm = lpips_table["CM"].idxmin()
            best_dps = lpips_table["DPS"].idxmin()
            print(f"- Best CM LPIPS at step {int(best_cm)}: {lpips_table.loc[best_cm, 'CM']:.4f}")
            print(f"- Best DPS LPIPS at step {int(best_dps)}: {lpips_table.loc[best_dps, 'DPS']:.4f}")
    if {"steps", "method", "missing_region_psnr"}.issubset(summary_df.columns):
        psnr_table = summary_df.pivot(index="steps", columns="method", values="missing_region_psnr")
        if {"CM", "DPS"}.issubset(psnr_table.columns):
            best_cm = psnr_table["CM"].idxmax()
            best_dps = psnr_table["DPS"].idxmax()
            print(f"- Best CM PSNR at step {int(best_cm)}: {psnr_table.loc[best_cm, 'CM']:.2f} dB")
            print(f"- Best DPS PSNR at step {int(best_dps)}: {psnr_table.loc[best_dps, 'DPS']:.2f} dB")
    print()
