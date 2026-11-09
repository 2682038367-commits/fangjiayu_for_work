"""Condition-scaled ordered-knowledge prompt; explicit replication assumption."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import read_cmapss, SETTING_COLUMNS
from .rebuilt_data import RebuiltWindowDataset, file_sha256
from .knowledge_prompt import build_knowledge_prompt, PROMPT_TEMPLATE_SHA256 as GLOBAL_TEMPLATE_SHA256

CONDITION_PROFILE = "condition_operating_sequence_v1"
CONDITION_TEMPLATE_SHA256 = hashlib.sha256((GLOBAL_TEMPLATE_SHA256 + file_sha256(Path(__file__))).encode()).hexdigest()


def build_condition_knowledge_prompt(window, settings, dataset):
    text = build_knowledge_prompt(window, settings, dataset)
    text = text.replace("14 globally Min-Max scaled sensors", "14 operating-condition-wise Min-Max scaled sensors")
    return text.replace("Observed context:", "Sensor scale: training-only per-condition Min-Max has corrected major operating-condition scale differences; do not interpret these as raw absolute sensor values. Observed context:", 1)


class ConditionKnowledgeWindowDataset(RebuiltWindowDataset):
    def __init__(self, data_dir, split, raw_data_dir):
        data_dir, raw_data_dir = Path(data_dir), Path(raw_data_dir)
        super().__init__(data_dir, split)
        manifest = json.loads((data_dir / "manifest.json").read_text())
        if manifest.get("normalization") != "condition_minmax":
            raise ValueError("condition knowledge requires condition_minmax")
        for filename, expected in manifest["raw_sha256"].items():
            if file_sha256(raw_data_dir / filename) != expected:
                raise ValueError("raw settings source changed")
        train, test, _ = read_cmapss(raw_data_dir, manifest["dataset"])
        raw = test if split == "test" else train
        engines = {int(u): f.sort_values("cycle").reset_index(drop=True) for u,f in raw.groupby("unit")}
        starts = np.load(data_dir / f"{split}_start_row.npy")
        self.prompts = []
        for i,(unit,start) in enumerate(zip(self.arrays["unit"], starts)):
            observed = engines[int(unit)].iloc[int(start):int(start)+40]
            if int(observed.iloc[-1].cycle) != int(self.arrays["cycle"][i]):
                raise ValueError("settings/window endpoint mismatch")
            self.prompts.append(build_condition_knowledge_prompt(self.arrays["x"][i], observed[SETTING_COLUMNS].to_numpy(), manifest["dataset"]))

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["prompt"] = self.prompts[index]
        return item
