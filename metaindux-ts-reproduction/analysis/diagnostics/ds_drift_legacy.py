from pathlib import Path
OUT = Path("results/frequency_threshold_validation")
def rd(p):
    with p.open() as f:
        h = f.readline().strip().split(",")
        return [dict(zip(h, l.strip().split(","))) for l in f if l.strip()]
new = rd(OUT/"fd002_fixed_025_seven_seeds_evaluation_raw.csv")
old = {(int(r["generation_seed"]), int(r["evaluator_seed"])): r
       for r in rd(OUT/"fd002_fixed_025_paired_comparison.csv")}
res = []
for r in new:
    k = (int(r["generation_seed"]), int(r["evaluator_seed"]))
    if k[0] in (13,23,43) and k in old:
        o = float(old[k]["discriminative_score_fixed"]); n = float(r["discriminative_score"])
        res.append((abs(n-o), abs(n-o)/abs(o)*100, k, o, n))
res.sort(reverse=True)
print("绝对偏差      相对%     格子(gen,eval)      旧DS        新DS")
for a, rel, k, o, n in res[:5]:
    print(f"{a:11.3e} {rel:8.2f}%  {str(k):>14}  {o:.6f}  {n:.6f}")
mx = res[0][1]
print(f"\n最大相对偏差 = {mx:.2f}%")
print("判定:", "PASS" if mx < 5 else ("可接受，但报告须标注测量噪声" if mx < 15 else "排查判别器"))
