#!/usr/bin/env python3
"""Recalculate C-MAPSS RMSE after clipping predictions to [0, 125]."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from rul_chronos.metrics import rmse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOWER = 0.0
UPPER = 125.0


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def metric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std_population": float(values.std(ddof=0)),
        "std_sample": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def recalculate(dataset: str, source_name: str, output_stem: str) -> dict[str, Any]:
    directory = PROJECT_ROOT / "artifacts" / dataset
    source_path = directory / source_name
    source = json.loads(source_path.read_text(encoding="utf-8"))
    rows = []
    for run in source["runs"]:
        prediction_path = PROJECT_ROOT / run["predictions"]
        prediction_document = json.loads(prediction_path.read_text(encoding="utf-8"))
        target = np.asarray(prediction_document["target"], dtype=np.float64)
        prediction = np.asarray(prediction_document["prediction"], dtype=np.float64)
        clipped = np.clip(prediction, LOWER, UPPER)
        raw_rmse = rmse(target, prediction)
        clipped_rmse = rmse(target, clipped)
        source_rmse_key = "test_rmse" if "test_rmse" in run else "max300_test_rmse"
        if not np.isclose(raw_rmse, float(run[source_rmse_key]), rtol=0.0, atol=1e-12):
            raise RuntimeError(f"Stored RMSE mismatch for {prediction_path}")
        rows.append({
            "seed": int(run["seed"]),
            "prediction_count": int(len(prediction)),
            "raw_prediction_min": float(prediction.min()),
            "raw_prediction_max": float(prediction.max()),
            "below_zero_count": int((prediction < LOWER).sum()),
            "above_125_count": int((prediction > UPPER).sum()),
            "total_clipped_count": int(((prediction < LOWER) | (prediction > UPPER)).sum()),
            "raw_test_rmse": float(raw_rmse),
            "clipped_test_rmse": float(clipped_rmse),
            "rmse_change": float(clipped_rmse - raw_rmse),
            "predictions": run["predictions"],
        })

    document = {
        "dataset": dataset,
        "source": str(source_path.relative_to(PROJECT_ROOT)),
        "clipping_range": [LOWER, UPPER],
        "note": "post-processing only; original predictions and reported metrics are unchanged",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "runs": rows,
        "aggregate": {
            "count": len(rows),
            "prediction_count": sum(row["prediction_count"] for row in rows),
            "below_zero_count": sum(row["below_zero_count"] for row in rows),
            "above_125_count": sum(row["above_125_count"] for row in rows),
            "total_clipped_count": sum(row["total_clipped_count"] for row in rows),
            "raw_test_rmse": metric_summary(rows, "raw_test_rmse"),
            "clipped_test_rmse": metric_summary(rows, "clipped_test_rmse"),
            "rmse_change": metric_summary(rows, "rmse_change"),
            "improved_seed_count": sum(row["rmse_change"] < 0 for row in rows),
            "unchanged_seed_count": sum(row["rmse_change"] == 0 for row in rows),
        },
    }
    json_path = directory / f"{output_stem}.json"
    csv_path = directory / f"{output_stem}.csv"
    atomic_write(json_path, json.dumps(document, indent=2) + "\n")
    fields = [
        "seed", "prediction_count", "raw_prediction_min", "raw_prediction_max",
        "below_zero_count", "above_125_count", "total_clipped_count",
        "raw_test_rmse", "clipped_test_rmse", "rmse_change", "predictions",
    ]
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    temporary.replace(csv_path)
    return document


def main() -> None:
    experiments = [
        ("FD001", "seed_results.json", "clipped_seed_results"),
        ("FD003", "seed_results.json", "clipped_seed_results"),
        ("FD003", "max300_results.json", "max300_clipped_results"),
    ]
    for dataset, source, output in experiments:
        result = recalculate(dataset, source, output)
        aggregate = result["aggregate"]
        print(
            f"{dataset}/{output}: runs={aggregate['count']} clipped={aggregate['total_clipped_count']}/"
            f"{aggregate['prediction_count']} raw_rmse={aggregate['raw_test_rmse']['mean']:.6f} "
            f"clipped_rmse={aggregate['clipped_test_rmse']['mean']:.6f} "
            f"change={aggregate['rmse_change']['mean']:+.6f}"
        )


if __name__ == "__main__":
    main()
