from __future__ import annotations
import warnings

# Silence Transformers' deprecation about old pytree registration API
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.utils\._pytree\._register_pytree_node is deprecated.*",
    category=UserWarning,
)

# Silence huggingface_hub FutureWarning about resume_download
warnings.filterwarnings(
    "ignore",
    message=r".*`resume_download` is deprecated.*",
    category=FutureWarning,
)

from pathlib import Path
from typing import Optional, List
import time
import typer
import yaml
from PIL import Image

from .utils.cameras import Pose
from .utils.image import make_contact_sheet  
from .utils.devices import pick_torch_device
from .zero123pp import Zero123PPBackend

app = typer.Typer(help="zero-mv — multi-view generation (Zero123++ backend).")

def _load_config(path: Optional[Path]) -> dict:
    """
    Load YAML config. If not provided, default to ./config.yaml when present.
    """
    cfg_path: Optional[Path] = path
    if cfg_path is None:
        default = Path("config.yaml")
        if default.exists():
            cfg_path = default
    if cfg_path and Path(cfg_path).exists():
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

def _echo_step(msg: str, color=typer.colors.BRIGHT_BLUE):
    typer.secho(f"[zero-mv] {msg}", fg=color)

def _echo_time(label: str, start: float, color=typer.colors.GREEN):
    dur = time.perf_counter() - start
    typer.secho(f"[zero-mv] {label}: {dur:.2f}s", fg=color)

@app.command()
def run(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to YAML config (defaults to ./config.yaml if present)"
    ),
    image: Optional[Path] = typer.Option(None, "--image", "-i", help="Input image"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output directory"),
    model_id: Optional[str] = typer.Option(
        None, "--model-id", help="HF model id (e.g. sudo-ai/zero123plus-v1.2)"
    ),
    steps: Optional[int] = typer.Option(None, "--steps", help="Inference steps (e.g. 36)"),
    grid: Optional[bool] = typer.Option(None, "--grid/--no-grid", help="Also save a contact sheet"),
    grid_cols: Optional[int] = typer.Option(None, "--grid-cols", help="Contact sheet columns"),
):
    """
    Run Zero123++ (in-process). The model outputs a fixed 6-view rig (v1.2).
    """
    t_all = time.perf_counter()
    t0 = time.perf_counter()
    cfg = _load_config(config)
    _echo_time("Config loaded", t0)

    def pick(name, default=None):
        val = locals().get(name, None)
        return val if val is not None else cfg.get(name, default)

    image    = pick("image")
    out      = Path(pick("out", Path("outputs/run")))
    model_id = pick("model_id", "sudo-ai/zero123plus-v1.2")
    steps    = int(pick("steps", 36))
    grid     = pick("grid", True)
    grid_cols= pick("grid_cols", None)

    if not image:
        raise typer.BadParameter("Provide --image or config.yaml:image")

    # Fixed 6-view rig per Zero123++ v1.2 (poses are not user-controllable).
    azimuths   = [30, 90, 150, 210, 270, 330]
    elevations = [20, -10, 20, -10, 20, -10]
    poses: List[Pose] = [Pose(yaw=az, pitch=el, fov=30.0) for az, el in zip(azimuths, elevations)]

        # Profiling: Torch import + device probe via shared helper
    _echo_step("Importing Torch and probing device…")
    t1 = time.perf_counter()
    import torch  # noqa: F401
    device = pick_torch_device(torch)
    _echo_time(f"Torch ready on {device}", t1)

    # Backend init (pass device explicitly to avoid re-probing in backend)
    _echo_step(f"Initializing Zero123++ pipeline: {model_id} (first run may download weights)…")
    t2 = time.perf_counter()
    backend = Zero123PPBackend(
        model_id=model_id,
        num_inference_steps=steps,
        device=device,
    )
    _echo_time("Zero123++ init complete", t2)

    # Run
    out.mkdir(parents=True, exist_ok=True)
    _echo_step("Running inference…")
    t3 = time.perf_counter()
    images = backend.render(Path(image), poses, out)
    _echo_time("Inference finished", t3)

    grid_path = None
    if grid and images:
        _echo_step("Creating contact sheet…")
        t4 = time.perf_counter()
        pil_imgs = [Image.open(p).convert("RGB") for p in images]
        cols = grid_cols if grid_cols else min(8, len(pil_imgs))
        sheet = make_contact_sheet(pil_imgs, cols=cols)
        grid_path = out / "contact_sheet.png"
        sheet.save(grid_path)
        _echo_time("Contact sheet saved", t4)

    # Summary
    _echo_time("Total elapsed", t_all, color=typer.colors.MAGENTA)
    typer.echo(f"[✓] Wrote {len(images)} images to: {out}")
    if grid_path:
        typer.echo(f"[✓] Contact sheet: {grid_path}")
    typer.echo("[i] Zero123++ (v1.2) returns a fixed 6-view rig; poses are not user-controllable.")

if __name__ == "__main__":
    app()
