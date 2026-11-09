"""Fig. 12-aligned prompt profile; unspecified serialization is an assumption."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from .rebuilt_data import RebuiltWindowDataset, file_sha256

PROFILE = "figure12_v1"
TASK_DESCRIPTION = (
    "C-MAPSS contains simulated aircraft turbofan operating records with sensor "
    "readings and operational settings. Estimate remaining useful life (RUL) "
    "in operating cycles from the observed engine condition."
)
DATASET_DESCRIPTION = (
    "Each sample is the latest observed 40-cycle window of 14 retained, "
    "Min-Max-normalized sensors. The operating settings are present in the "
    "original dataset but are not supplied in this prompt."
)
KNOWLEDGE_SHA256 = hashlib.sha256((TASK_DESCRIPTION + DATASET_DESCRIPTION).encode()).hexdigest()
PROMPT_TEMPLATE_SHA256 = file_sha256(Path(__file__))
SOURCE = "TS-MLLM paper, Fig. 12 (example prompt; exact generator unpublished)"


def build_figure12_prompt(window: np.ndarray | torch.Tensor) -> str:
    values = window.detach().cpu().numpy() if isinstance(window, torch.Tensor) else np.asarray(window)
    if values.shape != (40, 14) or not np.isfinite(values).all():
        raise ValueError("expected finite normalized window [40,14]")
    # Explicit serialization assumption: global scalar statistics over all
    # 40x14 values; trend from last-ten minus first-ten mean sensor values.
    start = float(values[:10].mean())
    end = float(values[-10:].mean())
    change = end - start
    trend = "upward" if change > 0.03 else "downward" if change < -0.03 else "stable"
    return (
        f"### Task Describe\n{TASK_DESCRIPTION}\n"
        "### Dataset Feature\n"
        "Analyze the preceding visual feature together with the following "
        "statistics from the current observed sensor window: "
        f"min={float(values.min()):.3f}, max={float(values.max()):.3f}, "
        f"median={float(np.median(values)):.3f}; "
        f"overall trend is {trend} (last-ten minus first-ten mean={change:+.3f}). "
        "Use observed evidence; do not assume a failure mode or unseen cycles.\n"
        f"### Dataset Describe\n{DATASET_DESCRIPTION}"
    )


class Figure12WindowDataset(RebuiltWindowDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["prompt"] = build_figure12_prompt(item["x"])
        return item
