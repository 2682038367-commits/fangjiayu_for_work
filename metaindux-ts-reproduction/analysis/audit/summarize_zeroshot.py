#!/usr/bin/env python3
"""Compare completed Table V evaluations with the published MetaIndux-TS arm."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    paper = json.loads((ROOT / 'docs/provenance/paper_table_v_zeroshot_metrics.json').read_text())
    rows = []
    for dataset, target in paper['metric']['values'].items():
        stem = dataset.lower() + '_table_v_evaluation'
        manifest_path = ROOT / 'results/zeroshot' / (stem + '_manifest.json')
        manifest = json.loads(manifest_path.read_text())
        expected = {(g, e) for g in manifest['config']['experiment']['generation_seeds']
                    for e in manifest['config']['experiment']['evaluator_seeds']}
        with (manifest_path.parent / (stem + '_raw.csv')).open() as handle:
            raw = list(csv.DictReader(handle))
        actual = [(int(r['generation_seed']), int(r['evaluator_seed'])) for r in raw]
        if manifest['status'] != 'complete' or len(actual) != len(expected) or set(actual) != expected:
            raise ValueError(f'incomplete or duplicated evaluations: {dataset}')
        with (manifest_path.parent / (stem + '_summary.csv')).open() as handle:
            aggregate = next(r for r in csv.DictReader(handle)
                             if r['metric'] == 'rmse' and r['generation_seed'] == 'aggregate')
        mean = float(aggregate['mean'])
        rows.append(dict(dataset=dataset, rmse=mean, paper_rmse=target,
                         gap=mean-target, relative_gap_pct=100*(mean-target)/target,
                         between_generation_std=float(aggregate['between_generation_std_of_means']),
                         total_std=float(aggregate['total_std']), manifest=str(manifest_path.relative_to(ROOT))))
    output = ROOT / 'results/zeroshot/table_v_comparison.csv'
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    print(f"Four-dataset mean RMSE: {sum(r['rmse'] for r in rows)/len(rows):.3f}; paper: 18.820")
    print('Numerical comparison only; assess fidelity and comparable seed variation before claiming reproduction.')


if __name__ == '__main__':
    main()
