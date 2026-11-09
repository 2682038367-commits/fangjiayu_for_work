"""Read-only verification of frozen main-result selection and source hashes."""
import json
from pathlib import Path
from ts_mllm.rebuilt_data import file_sha256


def main():
    root=Path(__file__).resolve().parents[1]
    lock=json.loads((root/"configs/main_protocol_v1.json").read_text())
    for ds,run in lock["runs"].items():
        for filename,expected in run["pinned_files"].items():
            if file_sha256(root/filename)!=expected:
                raise ValueError(f"pinned artifact changed: {filename}")
        r=json.loads((root/run["output_dir"]/"result.json").read_text())
        if r["dataset"]!=ds or r.get("prompt_profile","legacy")!="legacy":
            raise ValueError("wrong dataset/prompt selection")
        for key,value in lock["fixed_training"].items():
            if r[key]!=value:
                raise ValueError(f"fixed setting changed: {ds}/{key}")
        cfg=dict(r["model_config"]);cfg.setdefault("global_pooling","mean")
        if cfg!=run["model_config"]:
            raise ValueError("model configuration changed")
        expected_normalization="condition_minmax" if ds in ("FD002","FD004") else "global_minmax"
        if run["normalization"]!=expected_normalization:
            raise ValueError("wrong main normalization selection")
        print(f"{ds}: verified main selection; test={r['test']}",flush=True)


if __name__=="__main__":
    main()
