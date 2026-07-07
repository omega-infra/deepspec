import os
from pathlib import Path


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_symlink(src, dst):
    dst_path = Path(dst)
    if dst_path.is_symlink() or dst_path.exists():
        dst_path.unlink()
    dst_path.symlink_to(Path(src).resolve())


def resolve_model_path(name_or_path: str) -> str:
    """Check if a model exists locally before falling back to HuggingFace download.

    Lookup order:
      1. If ``name_or_path`` is already a local directory that exists → use it.
      2. If ``$DEEPSPEC_LOCAL_MODELS_DIR`` is set, check
         ``$DEEPSPEC_LOCAL_MODELS_DIR / <repo_name>``
         (e.g. ``/mnt/models/Qwen3-8B`` for ``Qwen/Qwen3-8B``).
      3. Otherwise return the original string so HF ``from_pretrained`` downloads it.
    """
    # 1. Direct local path
    if os.path.exists(name_or_path):
        return name_or_path

    # 2. Search in a configurable local models directory
    local_dir = os.environ.get("DEEPSPEC_LOCAL_MODELS_DIR")
    if local_dir:
        repo_name = name_or_path.split("/")[-1]
        candidate = os.path.join(local_dir, repo_name)
        if os.path.exists(candidate):
            return candidate

    # 3. Fall back to HuggingFace
    return name_or_path
