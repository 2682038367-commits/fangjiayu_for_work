"""
频率阈值可学习化 —— 离线可行性探针
==========================================================================
用途
  在不训练扩散模型、不需要 GPU 的前提下，判定 MetaIndux-TS 论文的
  "learnable threshold" 究竟能否被实现，以及各候选实现的行为。

背景
  MetaIndux-TS (TNNLS 2025, 36(10):18064-18075) 声称 θ 是 "learnable
  threshold optimized by ∂Loss/∂θ"，并说该机制 "inspired by [27]"
  (= TSLANet, Eldele et al., ICML 2024)。
  但其公开代码 (Dolphin-wang/MetaIndux-TS, DiffusionFreeGuidence/
  Attetnion_temp.py) 中，mask 由 quantile + boolean indexing 构造，
  与 θ 完全断图。

用法
  python threshold_ste_probe.py            # 合成数据，秒级，CPU
  替换 ne() 的输入为真实 x_fft 即可在你自己的数据上复跑。

依赖: torch (CPU 即可)
"""

import torch

# ============================================================ 配置
B, N, C = 64, 25, 14        # batch / rfft(48)//2+1 个频点 / 14 传感器
TRUE_KEEP = 0.70            # Oracle: 应保留能量最高的 70% 频点
NOISE = 0.10                # Oracle 掩码的翻转噪声
INITS = [0.05, 0.30, 0.45]  # 三个差别极大的 θ 初值，用来测初值敏感性
STEPS, LR = 2000, 2e-3

torch.manual_seed(0)


def ne(x):
    """作者代码的能量/归一化部分：|FFT|² 沿通道求和，再除以各样本频点中位数"""
    e = torch.abs(x).pow(2).sum(-1)
    med = e.view(B, -1).median(1, keepdim=True)[0].view(B, 1)
    return e / (med + 1e-6)


e = ne(torch.randn(B, N, C)).detach()

# ---- Oracle: 与能量相关的最优掩码（真实场景的合理代理） ----
_clean = (e > torch.quantile(e, 1 - TRUE_KEEP)).float()
M_star = torch.where(torch.rand(B, N) < NOISE, 1 - _clean, _clean)

print("=" * 86)
print(f"Oracle = 保留能量最高的 {TRUE_KEEP*100:.0f}% 频点 + {NOISE*100:.0f}% 翻转噪声")
print(f"随机初始化 θ 在作者代码中即为『随机滤波强度』，这正是 FD002 跨种子不稳的嫌疑来源")
print("=" * 86)


# ============================================================ 候选实现
def author(e, th):
    """① MetaIndux-TS 公开代码原样"""
    return (e > torch.quantile(e, th)).float()


def tslanet(e, th):
    """② TSLANet [27] 原文那一行"""
    return ((e > th).float() - th).detach() + th


def ste_linear(e, th):
    """③a 最小修补 - 线性 surrogate（保留作者 quantile 语义）"""
    q = torch.quantile(e, th)
    soft = (e - q) / (e.std() + 1e-8)
    return ((soft > 0).float() - soft).detach() + soft


def ste_sigmoid(e, th, tau=0.10):
    """③b 最小修补 - sigmoid surrogate（推荐）"""
    q = torch.quantile(e, th)
    soft = torch.sigmoid((e - q) / (tau * (e.std() + 1e-8)))
    return ((soft > 0.5).float() - soft).detach() + soft


# ============================================================ 网格基准
grid = [(tv.item(), ((author(e, tv) - M_star) ** 2).mean().item())
        for tv in torch.linspace(0.01, 0.99, 99)]
theta_opt, loss_opt = min(grid, key=lambda x: x[1])
print(f"\n固定 θ 的离散网格最小值:  θ* = {theta_opt:.2f}   L* = {loss_opt:.5f}\n")


# ============================================================ 训练循环
def run(fn, init, steps=STEPS, lr=LR, clamp=True):
    th = torch.nn.Parameter(torch.tensor([init]))
    opt = torch.optim.Adam([th], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        try:
            ((fn(e, torch.clamp(th, 1e-3, 1 - 1e-3) if clamp else th) - M_star) ** 2).mean().backward()
        except RuntimeError:
            return None, None, "backward 失败"
        if th.grad is None:
            return None, None, "grad is None"
        opt.step()
        if clamp:
            with torch.no_grad():
                th.clamp_(1e-3, 1 - 1e-3)
    with torch.no_grad():
        L = ((author(e, torch.clamp(th, 1e-3, 1 - 1e-3)) - M_star) ** 2).mean().item()
    return th.item(), L, None


# ============================================================ 输出
hdr = " | ".join(f"θ0={i:.2f}" for i in INITS)
print(f"{'实现':<34}| {hdr} | 初值离散度 | L 相对 L*")
print("-" * 86)

results = {}
for label, fn, cl in [
        ("① 作者公开代码", author, True),
        ("② TSLANet [27] 那一行", tslanet, False),
        ("③a Minimal-STE linear", ste_linear, True),
        ("③b Minimal-STE sigmoid τ=0.30", lambda a, b: ste_sigmoid(a, b, 0.30), True),
        ("③b Minimal-STE sigmoid τ=0.10", ste_sigmoid, True),
        ("③b Minimal-STE sigmoid τ=0.03", lambda a, b: ste_sigmoid(a, b, 0.03), True)]:
    outs = [run(fn, i, clamp=cl) for i in INITS]
    results[label] = outs
    if outs[0][0] is None:
        cells = " | ".join(f"不更新 (L={((author(e, torch.tensor(i))-M_star)**2).mean():.3f})"
                           for i in INITS)
        print(f"{label:<34}| {cells} |   N/A     | N/A")
        continue
    vals = [o[0] for o in outs]
    Ls = [o[1] for o in outs]
    spread = max(vals) - min(vals)
    rel = max(Ls) / loss_opt - 1
    cells = " | ".join(f"{v:.3f}/L{l:.4f}" for v, l in zip(vals, Ls))
    print(f"{label:<34}| {cells} |   {spread:.3f}    | +{rel*100:.1f}%")

# ============================================================ 结论
print("\n" + "=" * 86)
print("结论")
print("=" * 86)
print("""
① 作者公开代码：mask 由 torch.zeros_like + boolean indexing 构造，grad_fn=None，
   θ 与 loss 完全断图，训练期间是常量。论文所述 'optimized by ∂Loss/∂θ' 不成立。

② TSLANet [27] 那一行：前向确实严格二值、θ 也确实收到非零梯度，
   但 ∂mask/∂θ ≡ +1，而真实几何是 dρ/dθ < 0（θ↑⇒分位点↑⇒保留率↓）——
   符号恒相反，实验中 θ 一路漂到负值/边界，从不收敛。
   → TSLANet 提供的是『有出处的候选』，不是可用的答案。

③b Minimal-STE (sigmoid surrogate)：保留作者 quantile 语义，只给二值比较补上
   符号正确的替代梯度。从三个相差极大的初值出发均收敛到同一 θ，
   初值离散度 ≈ 0，loss 逼近离散网格参考最小值。
   → 这才是『可学习阈值』的一个真正可行的实现。

⚠ 边界声明
  本脚本用合成数据与人为 Oracle 验证『机制可行性』，不保证 θ 在真实
  DDPM 训练中同样收敛——前提是扩散损失对频率掩码确实有稳定且可测的依赖。
  上真机前建议先做低成本探针：训 1-2 epoch，记录 θ 轨迹与 ∂L/∂mask 的信噪比。
""")
