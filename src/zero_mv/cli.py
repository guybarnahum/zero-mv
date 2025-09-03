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
from typing import Optional, List, Dict, Any
import time
import typer
import yaml
from PIL import Image

from .utils.cameras import Pose
from .utils.image import make_contact_sheet
from .utils.devices import pick_torch_device
from .zero123pp import Zero123PPBackend

app = typer.Typer(help="zero-mv — multi-view generation (Zero123++ backend).")

def _load_config(path: Optional[Path]) -> Dict[str, Any]:
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

def _merge_config(base: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    """
    Merge CLI args over config values. Only non-None CLI values win.
    """
    out = dict(base) if base else {}
    for k, v in overrides.items():
        if v is not None:
            out[k] = v
    return out

def _echo_step(msg: str, color=typer.colors.BRIGHT_BLUE):
    typer.secho(f"[zero-mv] {msg}", fg=color)

def _echo_time(label: str, start: float, color=typer.colors.GREEN):
    dur = time.perf_counter() - start
    typer.secho(f"[zero-mv] {label}: {dur:.2f}s", fg=color)

def _rename_if_exists(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            dst.unlink()
        except Exception:
            pass
    src.rename(dst)

@app.command()
def run(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to YAML config (defaults to ./config.yaml if present)"
    ),
    image: Optional[Path] = typer.Option(None, "--image", "-i", help="Input image"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output directory root"),
    model_id: Optional[str] = typer.Option(
        None, "--model-id", help="HF model id (e.g. sudo-ai/zero123plus-v1.2)"
    ),
    steps: Optional[int] = typer.Option(None, "--steps", help="Inference steps (e.g. 36)"),
    grid: Optional[bool] = typer.Option(None, "--grid/--no-grid", help="Also save a contact sheet"),
    grid_cols: Optional[int] = typer.Option(None, "--grid-cols", help="Contact sheet columns"),
):
    """
    Run Zero123++ (in-process). The model outputs a fixed 6-view rig (v1.2).

    NEW:
      - Args override config.yaml values (even when --config is used).
      - Outputs go under {out}/{basename(image)}/
      - File names: 000_{base}_view.png, ..., {base}_views_grid.png, {base}_views_sheet.png
    """
    t_all = time.perf_counter()

    # Load config
    t0 = time.perf_counter()
    cfg = _load_config(config)
    _echo_time("Config loaded", t0)

    # Merge: CLI args (non-None) override config
    merged = _merge_config(
        cfg,
        image=str(image) if image is not None else None,
        out=str(out) if out is not None else None,
        model_id=model_id,
        steps=steps,
        grid=grid,
        grid_cols=grid_cols,
    )

    # Extract final settings with defaults
    image_path = merged.get("image", None)
    if not image_path:
        raise typer.BadParameter("Provide --image or config.yaml:image")
    image_path = Path(image_path)

    out_root = Path(merged.get("out", "outputs"))  # root folder
    base = image_path.stem
    run_dir = out_root / base  # e.g., outputs/person
    run_dir.mkdir(parents=True, exist_ok=True)

    model_id = merged.get("model_id", "sudo-ai/zero123plus-v1.2")
    steps = int(merged.get("steps", 36))
    grid = bool(merged.get("grid", True))
    grid_cols = merged.get("grid_cols", None)

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

    # Run (backend writes: zpp_grid.png + 000_zpp_view.png..005_zpp_view.png in run_dir)
    _echo_step("Running inference…")
    t3 = time.perf_counter()
    tile_paths = backend.render(image_path, poses, run_dir)
    _echo_time("Inference finished", t3)

    # Rename/massage filenames to requested scheme:
    #   000_{base}_view.png ... 005_{base}_view.png
    final_tiles: List[Path] = []
    for i, p in enumerate(sorted(tile_paths)):
        dst = run_dir / f"{i:03d}_{base}_view.png"
        if p.name != dst.name:
            _rename_if_exists(p, dst)
        else:
            dst = p
        final_tiles.append(dst)

    # Rename model grid to {base}_views_grid.png (if backend created it)
    model_grid = run_dir / "zpp_grid.png"
    final_grid_path = None
    if model_grid.exists():
        final_grid_path = run_dir / f"{base}_views_grid.png"
        if model_grid.name != final_grid_path.name:
            _rename_if_exists(model_grid, final_grid_path)
        else:
            final_grid_path = model_grid

    # Optional contact sheet: {base}_views_sheet.png
    sheet_path = None
    if grid and final_tiles:
        _echo_step("Creating contact sheet…")
        t4 = time.perf_counter()
        pil_imgs = [Image.open(p).convert("RGB") for p in final_tiles]
        cols = grid_cols if grid_cols else min(8, len(pil_imgs))
        sheet = make_contact_sheet(pil_imgs, cols=cols)
        sheet_path = run_dir / f"{base}_views_sheet.png"
        sheet.save(sheet_path)
        _echo_time("Contact sheet saved", t4)

    # Summary
    _echo_time("Total elapsed", t_all, color=typer.colors.MAGENTA)
    typer.echo(f"[✓] Wrote {len(final_tiles)} images to: {run_dir}")
    if final_grid_path:
        typer.echo(f"[✓] Model grid: {final_grid_path}")
    if sheet_path:
        typer.echo(f"[✓] Contact sheet: {sheet_path}")
    typer.echo("[i] Zero123++ (v1.2) returns a fixed 6-view rig; poses are not user-controllable.")

if __name__ == "__main__":
    app()
