from __future__ import annotations
from pathlib import Path
from typing import Sequence, Optional, List

from .utils.image import load_image  # save_image/annotate not needed here
from .utils.image import to_square, try_split_grid
from .utils.cameras import Pose


def _pick_device() -> str:
    try:
        import torch  # noqa: F401
        import torch.backends  # noqa: F401

        import torch  # reimport for clarity
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class Zero123PPBackend:
    """
    Monolithic (in-process) Zero123++ runner using the Diffusers custom pipeline.

    Args:
      model_id: HF repo id (default: "sudo-ai/zero123plus-v1.2")
      dtype: "auto" | "fp16" | "fp32"  (auto = fp16 on cuda/mps, else fp32)
      scheduler: Optional scheduler name override ("EulerAncestralDiscrete" supported)
      num_inference_steps: e.g., 28–36 for speed/quality balance; 75–100 for finer detail

    Notes:
      - Zero123++ v1.2 outputs a FIXED set of 6 views.
      - We save the combined grid (zpp_grid.png) and split tiles
        (000_zpp_view.png ... 005_zpp_view.png) directly under the run directory.
    """

    def __init__(
        self,
        model_id: str = "sudo-ai/zero123plus-v1.2",
        dtype: str = "auto",
        scheduler: Optional[str] = "EulerAncestralDiscrete",
        num_inference_steps: int = 36,
    ):
        self.model_id = model_id
        self.num_inference_steps = int(num_inference_steps)
        self.device = _pick_device()

        # Heavy imports delayed until instantiation
        import torch
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

        # Choose dtype
        if dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype == "fp32":
            torch_dtype = torch.float32
        else:  # auto
            torch_dtype = torch.float16 if self.device in {"cuda", "mps"} else torch.float32

        # Load custom pipeline (documented in the Zero123++ repo)
        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_id,
            custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch_dtype,
        )

        # Optional scheduler override (Euler Ancestral with trailing spacing)
        if scheduler in {"EulerAncestralDiscrete", "euler_ancestral"}:
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipe.scheduler.config, timestep_spacing="trailing"
            )

        self.pipe.to(self.device)

        # (Optional) memory savers; harmless if unsupported
        try:
            self.pipe.enable_attention_slicing()
            if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
                self.pipe.vae.enable_slicing()
                self.pipe.vae.enable_tiling()
        except Exception:
            pass

    def _run_zero123pp(self, image_path: Path, out_dir: Path) -> tuple[List[Path], Path]:
        """
        Run the pipeline once and save results directly in out_dir:
          - zpp_grid.png  (combined grid from the model)
          - 000_zpp_view.png ... 005_zpp_view.png  (split tiles if a 6-tile grid is detected)

        Returns (tile_paths, grid_path).
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        cond = load_image(image_path)
        cond_sq = to_square(cond, min_side=320)

        # Inference
        result = self.pipe(cond_sq, num_inference_steps=self.num_inference_steps).images[0]  # PIL.Image

        grid_path = out_dir / "zpp_grid.png"
        result.save(grid_path)

        tiles = try_split_grid(result)
        tile_paths: List[Path] = []
        for i, tile in enumerate(tiles):
            p = out_dir / f"{i:03d}_zpp_view.png"
            tile.save(p)
            tile_paths.append(p)

        return tile_paths, grid_path

    def render(self, image_path: str | Path, poses: Sequence[Pose], run_dir: str | Path) -> List[Path]:
        """
        Generate the fixed 6-view set. `poses` is ignored (kept for API compatibility).

        Saves (flat layout in run_dir):
          - zpp_grid.png
          - 000_zpp_view.png ... 005_zpp_view.png

        Returns the list of tile paths (000..005).
        """
        image_path = Path(image_path)
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        tile_paths, _ = self._run_zero123pp(image_path, run_dir)
        return tile_paths
