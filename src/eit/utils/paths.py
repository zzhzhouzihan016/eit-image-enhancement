from pathlib import Path
from typing import Union


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
DOCS_DIR = PROJECT_ROOT / "docs"


def resolve_from_root(path_like: Union[str, Path]) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_parent(path_like: Union[str, Path]) -> Path:
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
