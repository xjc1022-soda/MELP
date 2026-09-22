"""Filesystem roots for datasets, preprocessing output and external weights.

Every root resolves from the environment with a built-in default, so the data tree
can move (or differ per machine) without editing any call site:

    MELP_RAW_DATA_PATH        raw datasets, one directory per dataset:
                              RAW_DATA_PATH/{mimic-iv-ecg,ptbxl,icbeb,chapman,code}
    MELP_PROCESSED_DATA_PATH  output of scripts/preprocess/* (denoised records,
                              preprocessed reports); defaults next to the raw root
    MELP_SPLIT_DIR            data splits; defaults to the ones shipped in this repo.
                              The MIMIC-IV-ECG splits are NOT shipped (they are a
                              separate download), so point this at a directory holding
                              them to pretrain without copying them into the checkout
    MELP_ECGFM_PATH           ECG-FM (wav2vec2-CMSC) Lightning checkpoint used to
                              initialise the ECG encoder; empty means "train from
                              scratch", which is what the models check for

Repo-internal assets (data splits, prompts, logs) are addressed relative to the
checkout and travel with it.
"""
import os
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[2]


def _root(var: str, default: Path) -> Path:
    """Return ``$var`` as a path if set and non-empty, else ``default``."""
    value = os.environ.get(var, "").strip()
    return Path(value).expanduser() if value else default


RAW_DATA_PATH = _root("MELP_RAW_DATA_PATH", Path("/disk1/jcxu/ECG/raw"))
PROCESSED_DATA_PATH = _root(
    "MELP_PROCESSED_DATA_PATH", RAW_DATA_PATH.parent / "processed_data"
)
# Kept a plain string: the models test it for truthiness to decide whether to load
# pretrained ECG encoder weights, and Path("") would be PosixPath(".") -> truthy.
ECGFM_PATH = os.environ.get("MELP_ECGFM_PATH", "").strip()

SPLIT_DIR = _root("MELP_SPLIT_DIR", ROOT_PATH / "src/melp/data_split")
PROMPT_PATH = ROOT_PATH / "src/melp/prompt/CKEPE_prompt.json"
DATASET_LABELS_PATH = ROOT_PATH / "src/melp/prompt/dataset_class_names.json"
RESULTS_PATH = ROOT_PATH / "logs/melp/results"


if __name__ == "__main__":
    for name in ("ROOT_PATH", "RAW_DATA_PATH", "PROCESSED_DATA_PATH", "ECGFM_PATH",
                 "SPLIT_DIR", "PROMPT_PATH", "DATASET_LABELS_PATH", "RESULTS_PATH"):
        value = globals()[name]
        exists = Path(value).exists() if str(value) else False
        print(f"{name:22s} {str(value) or '<unset>':50s} {'ok' if exists else 'MISSING'}")
