from pathlib import Path
import csv
OUT = Path("results/frequency_threshold_validation")
def rd(p):
    with p.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))
fx = {(int(r["generation_seed"]), int(r["evaluator_seed"])): r
      for r in rd(OUT/"fd002_fixed_025_seven_seeds_evaluation_raw.csv")}
pb = {(int(r["generation_seed"]), int(r["evaluator_seed"])): r
      for r in rd(OUT/"fd002_public_arm_seven_seeds_evaluation_raw.csv")}
keys = sorted(set(fx) & set(pb))
assert len(keys) == 35, f"配对行数 {len(keys)} != 35，贴回排查"
with (OUT/"fd002_seven_seeds_paired.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["generation_seed","evaluator_seed","rmse_fixed","rmse_random",
                "discriminative_score_fixed","discriminative_score_random",
                "mae_fixed","mae_random","rul_score_fixed","rul_score_random"])
    for k in keys:
        a, b = fx[k], pb[k]
        w.writerow([k[0],k[1],a["rmse"],b["rmse"],
                    a["discriminative_score"],b["discriminative_score"],
                    a["mae"],b["mae"],a["rul_score"],b["rul_score"]])
print(f"写出 fd002_seven_seeds_paired.csv  {len(keys)} 行")
