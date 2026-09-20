"""Source-backed knowledge plus observed settings, not the author's prompt."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import SETTING_COLUMNS, read_cmapss
from .qwen_multimodal import build_dynamic_prompt
from .rebuilt_data import RebuiltWindowDataset, file_sha256

PROFILE = "operating_sequence_v2"
KNOWLEDGE = ("Three operating settings affect engine performance. Initial wear and manufacturing "
             "variation differ between engines; sensor noise is present. Faults develop and grow "
             "during operation. Reasoning guidance: a setting-related sensor shift alone is not "
             "proof of degradation; interpret persistent multi-sensor changes with operating context.")
KNOWLEDGE_SHA256 = hashlib.sha256(KNOWLEDGE.encode()).hexdigest()
SOURCE = "https://catalog.data.gov/dataset/cmapss-jet-engine-simulated-data"
PROMPT_TEMPLATE_SHA256 = file_sha256(Path(__file__))

def build_knowledge_prompt(window, settings, dataset):
    settings = np.asarray(settings, dtype=np.float64)
    if settings.ndim != 2 or settings.shape[1] != 3 or not 1 <= len(settings) <= 40 or not np.isfinite(settings).all():
        raise ValueError("expected finite observed settings [1..40,3]")
    rounded = np.column_stack((settings[:, 0].round(0), settings[:, 1].round(2), settings[:, 2].round(0)))
    groups, inverse, counts = np.unique(rounded, axis=0, return_inverse=True, return_counts=True)
    switches = int(np.any(rounded[1:] != rounded[:-1], axis=1).sum())
    # Lossless ordering of the already-rounded setting tuples; not a fitted
    # condition classifier. Dictionary codes keep all observed cycles in budget.
    codes = [chr(65+i) for i in range(len(groups))]
    summary = "; ".join(f"{code}=({g[0]:g},{g[1]:g},{g[2]:g}):{c}/{len(settings)}" for code, g, c in zip(codes, groups, counts))
    sequence = " ".join(codes[i] for i in inverse)
    left_pad = 40-len(settings)
    sensors = build_dynamic_prompt(window).split("Global sensor std=", 1)[1].split("Current sensor means:", 1)[0].strip()
    return (f"Dataset description: C-MAPSS {dataset}, turbofan monitoring; "
            f"{'six' if dataset in ('FD002','FD004') else 'one'} operating conditions; "
            "40-cycle window, 14 globally Min-Max scaled sensors.\n"
            f"###Domain: {KNOWLEDGE}\n"
            f"Observed context: rounded setting tuples (setting1,setting2,setting3) and counts: {summary}. "
            f"Ordered observed-cycle codes, oldest to newest: {sequence}. "
            f"Code j maps to sensor row {left_pad}+j (1-based); first {left_pad} rows are padding. "
            f"Setting-group switches={switches}; latest raw settings="
            f"({settings[-1,0]:.4f},{settings[-1,1]:.4f},{settings[-1,2]:.1f}). "
            f"Sensor summary: global sensor std={sensors}\n"
            "###Instruction: Estimate remaining operating cycles from observed evidence; "
            "do not equate operating-condition changes with faults.")

class KnowledgeWindowDataset(RebuiltWindowDataset):
    def __init__(self, data_dir, split, raw_data_dir):
        super().__init__(data_dir, split)
        manifest = json.loads((data_dir / "manifest.json").read_text())
        if manifest.get("normalization", "global_minmax") != "global_minmax":
            raise ValueError("knowledge profile requires global Min-Max")
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
            self.prompts.append(build_knowledge_prompt(self.arrays["x"][i], observed[SETTING_COLUMNS].to_numpy(), manifest["dataset"]))

    def __getitem__(self,index):
        item = super().__getitem__(index)
        item["prompt"] = self.prompts[index]
        return item
