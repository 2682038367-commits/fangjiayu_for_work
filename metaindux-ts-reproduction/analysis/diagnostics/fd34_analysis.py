import numpy as np
from itertools import permutations

def spearman(x, y):
    rx = np.argsort(np.argsort(x)) + 1
    ry = np.argsort(np.argsort(y)) + 1
    d2 = ((rx - ry) ** 2).sum()
    n = len(x)
    rho = 1 - 6 * d2 / (n * (n * n - 1))
    # 精确置换检验双尾 p
    cnt = 0; tot = 0
    obs = abs(rho)
    for p in permutations(range(1, n + 1)):
        rr = np.array(p); dd2 = ((rx - rr) ** 2).sum()
        r = 1 - 6 * dd2 / (n * (n * n - 1))
        tot += 1
        if abs(r) >= obs - 1e-12: cnt += 1
    return rx, ry, rho, cnt / tot

datasets = {
    "FD002": dict(rmse=[26.990,23.540,26.849,25.213,31.366,35.304,28.799],
                  ds=[0.1306,0.1067,0.4563,0.0988,0.4211,0.4988,0.1221],
                  seeds=[3,13,23,33,43,53,63], paper=(24.565,0.109)),
    "FD003": dict(rmse=[20.410,18.833,21.965,20.781,21.597],
                  ds=[0.235,0.123,0.168,0.146,0.286],
                  seeds=[3,13,23,33,43], paper=(16.206,0.114)),
    "FD004": dict(rmse=[35.733,30.177,36.207,33.720,30.547],
                  ds=[0.289,0.465,0.241,0.458,0.473],
                  seeds=[3,13,23,33,43], paper=(26.577,0.239)),
}

print("以下为本地生成数据结果的描述统计；不与论文点估计做显著性比较。")
print(f"{'数据集':<8}{'RMSE均值':>10}{'SD':>8}{'SEM':>8}")
for name, d in datasets.items():
    r = np.array(d["rmse"])
    m, sd = r.mean(), r.std(ddof=1); sem = sd/np.sqrt(len(r))
    print(f"{name:<8}{m:>10.3f}{sd:>8.3f}{sem:>8.3f}")
print()
print(f"{'数据集':<8}{'DS均值':>10}{'SD':>8}{'SEM':>8}")
for name, d in datasets.items():
    s = np.array(d["ds"])
    m, sd = s.mean(), s.std(ddof=1); sem = sd/np.sqrt(len(s))
    print(f"{name:<8}{m:>10.4f}{sd:>8.4f}{sem:>8.4f}")
print()
print("=== RMSE 与 判别分数的秩相关 ===")
for name, d in datasets.items():
    rx, ry, rho, pv = spearman(d["rmse"], d["ds"])
    print(f"{name}: Spearman rho = {rho:+.3f}  置换检验双尾 p = {pv:.4f}")
    print(f"   RMSE 序 {dict(zip(d['seeds'], rx.tolist()))}")
    print(f"   DS   序 {dict(zip(d['seeds'], ry.tolist()))}")

print()
print("=== 方差分解：跨种子 SD 中有多少是真实的生成差异 ===")
# FD003 RMSE: pooled SD 1.578, n_eval=5 ; 跨种子 SD 1.223
for label, pooled, cross, n_eval in [("FD003 RMSE", 1.578, 1.223, 5),
                                     ("FD004 DS", 0.024, 0.111, 5)]:
    var_obs = cross ** 2
    var_err = pooled ** 2 / n_eval
    var_true = max(var_obs - var_err, 0.0)
    print(f"{label}: 观测方差 {var_obs:.5f} = 抽样误差 {var_err:.5f} + 真实种子差异 {var_true:.5f}")
    print(f"   真实种子间 SD ≈ {np.sqrt(var_true):.3f}，占观测方差 {100*var_true/var_obs:.0f}%")
