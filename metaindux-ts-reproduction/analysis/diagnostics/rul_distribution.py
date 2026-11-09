"""Audit observed real and generated RUL labels in the zero-shot w48 arm."""
from pathlib import Path
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/zeroshot/rul_distribution'
SEEDS = [3, 13, 23, 33, 43]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / 'upstream'))
    from data.CMAPSSDataset import CMAPSSDataset
    rows, exact, audits = [], [], {}
    edges = [-0.5, 10.5, 20.5, 30.5, 60.5, 90.5, 124.5, 125.5]
    names = ['0–10', '11–20', '21–30', '31–60', '61–90', '91–124', '125 (capped)']
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    previous = Path.cwd()
    try:
        os.chdir(ROOT / 'upstream')
        for ds, ax in zip(['FD001', 'FD002', 'FD003', 'FD004'], axes.flat):
            dataset = CMAPSSDataset(ds, 48, [1000])
            real = dataset.get_label_slice(dataset.get_train_data()).numpy().ravel()
            # Same terminal labels used by get_last_data_slice, without loading features.
            test_frame = dataset.get_test_data().groupby('id', sort=True).tail(1)
            test = np.minimum(test_frame['RUL'].to_numpy(), 125)
            generated = []
            identical = {}
            for seed in SEEDS:
                with np.load(ROOT / f'outputs/{ds}_w48_seed{seed}_zeroshot_paper.npz') as z:
                    labels = z['label'].ravel().copy()
                identical[str(seed)] = bool(np.array_equal(real, labels))
                assert len(real) == len(labels) and np.isfinite(labels).all()
                generated.append(labels)
            audits[ds] = {'every_seed_labels_equal_real_in_order': identical}
            sources = {'real_train_windows': real,
                       'generator_training_subset': real[real > 30],
                       'generated_seed3': generated[0],
                       'generated_all_5_seeds': np.concatenate(generated),
                       'real_test_terminal': test,
                       'real_test_fatigue_evaluation': test[test <= 30]}
            for source, values in sources.items():
                assert np.all((values >= 0) & (values <= 125))
                counts = np.bincount(values.astype(int), minlength=126)
                row = dict(dataset=ds, source=source, count=len(values),
                           minimum=float(values.min()), maximum=float(values.max()),
                           mean=float(values.mean()), median=float(np.median(values)),
                           rul_le30=int((values <= 30).sum()),
                           rul_gt30=int((values > 30).sum()), rul_eq125=int((values == 125).sum()),
                           pct_le30=float(np.mean(values <= 30)*100),
                           pct_gt30=float(np.mean(values > 30)*100),
                           pct_eq125=float(np.mean(values == 125)*100))
                rows.append(row)
                for rul, count in enumerate(counts):
                    exact.append(dict(dataset=ds, source=source, rul=rul,
                                      count=int(count), percent=float(count/len(values)*100)))
            real_pct = np.histogram(real, bins=edges)[0] / len(real) * 100
            gen_pct = np.histogram(generated[0], bins=edges)[0] / len(generated[0]) * 100
            x = np.arange(len(names))
            ax.bar(x-.2, real_pct, width=.4, label='Real training windows', color='#3876ad')
            ax.bar(x+.2, gen_pct, width=.4, label='Generated windows (each seed)', color='#e8973f')
            ax.set_title(f'{ds}: {len(real):,} windows per dataset')
            ax.set_xticks(x, names, rotation=25, ha='right')
            ax.set_ylabel('Windows (%)')
            ax.grid(axis='y', alpha=.2)
            ax.legend(fontsize=8)
    finally:
        os.chdir(previous)
    fig.suptitle('RUL label distributions: real vs zero-shot generated (window = 48)\n'
                 'RUL is the window-end label; values above 125 are clipped to 125', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(OUT / 'real_vs_generated_rul.png', dpi=180)
    fig.savefig(OUT / 'real_vs_generated_rul.pdf')
    plt.close(fig)
    for filename, data in [('summary.csv', rows), ('exact_rul_counts.csv', exact)]:
        with (OUT / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (OUT / 'label_equality_audit.json').write_text(json.dumps(audits, indent=2)+'\n')
    print(json.dumps([r for r in rows if r['source'] == 'real_train_windows'], indent=2))
    print('Every generated seed preserves real labels:',
          all(all(v['every_seed_labels_equal_real_in_order'].values()) for v in audits.values()))


if __name__ == '__main__':
    main()
