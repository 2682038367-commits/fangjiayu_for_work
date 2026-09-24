#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
固定臂（freq_threshold_025）补跑：配方发现 + 完成后 θ 校验

背景
----
FD002 / 窗口 48 的固定 θ=0.25 臂目前只有种子 13 / 23 / 43（n=3, 配对 p=0.39），
不足以支撑任何结论。本脚本用于把该臂补到 7 个种子（补 3 / 33 / 53 / 63）。

它做两件事，互不依赖：

  【A】--discover（默认）  只读扫描。反推当年固定臂的启动配方：
       1) 全文检索项目内的 py / sh / yaml / json / md / log，找出提到
          freq_threshold_025 / frequency_threshold_init 等关键字的文件与行 ——
          命中处几乎必然就是当年生成 checkpoint 的那个脚本；
       2) 列出所有带 argparse "--seed" 且涉及 threshold 参数的候选入口，
          并打印它们的 threshold 相关参数名；
       3) 打印 results/frequency_threshold_validation/ 下已有产物（含 json 内容），
          从 metadata 里回收超参，避免照着记忆重写导致口径漂移；
       4) 根据以上信息拼出 4 条待补命令的草稿（dry-run，不执行）。

  【B】--verify            训练完成后校验。读 checkpoints/FD002_w48_seed*_
       freq_threshold_025.pth，确认两个频率模块的 threshold_param 严格等于 0.25，
       防止"跑了半天其实没固定住"。需要 torch，请在有 torch 的解释器下运行。

用法（在项目根目录执行）
------------------------
    .venv/bin/python backfill_fixed_arm.py --discover
    .venv/bin/python backfill_fixed_arm.py --verify

可选：
    --root PATH        项目根目录，默认当前目录
    --two-column       以 patched 模式的 yes 输出一列相似度
    --max-hits N       每段最多打印多少条命中（默认 25）

设计约束
--------
* 只用标准库做扫描，任何 python3 都能跑 --discover；
  只有 --verify 需要 torch。
* 全程只读，不写任何文件，不改仓库内容。
* 跳过 .venv / checkpoints / outputs 等目录，避免读上 GB 的权重文件。
"""

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------

TEXT_EXT = {
    ".py", ".sh", ".bash", ".yaml", ".yml", ".json", ".jsonl",
    ".toml", ".md", ".log", ".txt", ".cfg", ".ini",
}

# 这些目录体积大或纯二进制，跳过。logs 保留——里面常写着真正的命令行。
EXCLUDE_DIRS = {
    ".venv", "venv", "__pycache__", ".git", "node_modules",
    "checkpoints", "outputs", ".ipynb_checkpoints", "wandb",
}

MAX_FILE_SIZE = 2_000_000  # 2 MB

# 关键字按优先级分组：第一组一旦命中就能直接锁定脚本
KEY_STRONG = [
    "freq_threshold_025",
    "freq-threshold-025",
    "frequency_threshold_025",
    "frequency-threshold-025",
    "freq_threshold_0.25",
]
KEY_MEDIUM = [
    "frequency_threshold_init",
    "freq_threshold_init",
    "frequency_threshold",
    "freq_threshold",
    "threshold_param",
]
KEY_WEAK = [
    "fixed_025",
    "threshold_025",
    "fixed_threshold",
]

SEP = "=" * 88


def hr(title):
    print("\n" + SEP)
    print(title)
    print(SEP)


# --------------------------------------------------------------------------
# 扫描工具
# --------------------------------------------------------------------------

def iter_text_files(root, max_size=MAX_FILE_SIZE):
    """遍历可读的文本文件，跳过体积过大的与排除目录。"""
    self_name = Path(__file__).name  # 别把自己搜出来
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if fn == self_name:
                continue
            p = Path(dirpath) / fn
            if p.suffix.lower() not in TEXT_EXT:
                continue
            try:
                if p.stat().st_size > max_size:
                    continue
            except OSError:
                continue
            yield p


def grep(root, keywords, max_hits, max_line=200):
    """返回 [(file, lineno, line, keyword)]，按文件排序。"""
    pats = [(k, re.compile(re.escape(k), re.IGNORECASE)) for k in keywords]
    hits = []
    files_scanned = 0
    for p in iter_text_files(root):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        files_scanned += 1
        for i, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if not s or len(s) > max_line:
                continue
            for kw, pat in pats:
                if pat.search(s):
                    hits.append((str(p), i, s, kw))
                    break  # 一行只记一次，由最强关键字负责
    hits.sort(key=lambda t: (t[0], t[1]))
    return hits[:max_hits], files_scanned


def rel(root, p):
    try:
        return str(Path(p).relative_to(root))
    except ValueError:
        return str(p)


# --------------------------------------------------------------------------
# 候选入口识别
# --------------------------------------------------------------------------

ARG_SIG = re.compile(r"add_argument\(\s*[\"'](--?[A-Za-z0-9_\-]+)")


def scan_entrypoints(root):
    """找出所有带 --seed 的 argparse 入口，附带其 threshold 相关参数。"""
    cands = []
    for p in iter_text_files(root):
        if p.suffix != ".py":
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "add_argument" not in text:
            continue
        args = set(ARG_SIG.findall(text))
        if not any(a.lstrip("-") in ("seed", "seed_id", "random_seed") for a in args):
            continue
        th_args = sorted(a for a in args
                         if "thresh" in a.lower() or "freq" in a.lower() or "theta" in a.lower())
        flag_args = sorted(a for a in args
                           if any(x in a.lower() for x in ("tag", "exp", "name", "suffix", "run")))
        cands.append({
            "path": str(p),
            "n_args": len(args),
            "seed_args": sorted(a for a in args if "seed" in a.lower()),
            "th_args": th_args,
            "flag_args": flag_args,
        })
    # 有 threshold 参数的排前面
    cands.sort(key=lambda c: (-(len(c["th_args"]) > 0), c["path"]))
    return cands


# --------------------------------------------------------------------------
# 已有固定臂产物
# --------------------------------------------------------------------------

def dump_existing(root, max_hits):
    """列出固定臂已有产物，并尽量把 metadata json 的内容打出来。"""
    fixed_dirs = []
    hit_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        low = dirpath.lower()
        if "frequency_threshold_validation" in low or "fixed_025" in low:
            fixed_dirs.append(dirpath)
        for fn in filenames:
            f_low = fn.lower()
            if ("025" in f_low or "threshold" in f_low) and Path(fn).suffix.lower() in TEXT_EXT:
                hit_files.append(str(Path(dirpath) / fn))
    return sorted(set(fixed_dirs)), sorted(set(hit_files))[:max_hits]


def show_json(path, limit=2400):
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8", errors="ignore"))
    except Exception as e:
        print(f"    [无法解析为 json] {e}")
        return
    txt = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False)
    print(textwrap.indent(txt[:limit], "    "))
    if len(txt) > limit:
        print(f"    ... 已截断（共 {len(txt)} 字符）")


# --------------------------------------------------------------------------
# 命令草稿
# --------------------------------------------------------------------------

SEEDS_TODO = [3, 33, 53, 63]
SEEDS_DONE = [13, 23, 43]


def emit_plan(root, cands, strong_hits):
    print()
    print("判定规则：")
    print("  1. 若下方【T1】直接命中 freq_threshold_025 字面量 → 该文件就是当年入口，")
    print("     照抄它所在的那一行命令，只把 seed 换成 3/33/53/63。")
    print("  2. 若只在【T2】见到 argparse 参数 → 用【候选入口】里的脚本 + 参数名手动拼。")
    print("  3. 两者都空 → 固定臂多半是手写命令跑的，翻 shell history / 直接给我下面的输出。")

    strong_unlocks = bool(strong_hits)
    print()
    if strong_unlocks:
        print(">>> 已锁定入口候选，请重点看下面命中的那几行，把整条命令抄出来：")
        for f, ln, line, kw in strong_hits[:10]:
            print(f"    {rel(root, f)}:{ln}  [{kw}]")
    else:
        print(">>> 未在项目文件中发现 'freq_threshold_025' 字面量。")
        print("    可能情况：checkpoint 名由你在命令行临时指定，因此源码里搜不到。")
        print("    此时应以【候选入口】+ shell history 为准。")

    # 生成草稿
    primary = None
    for c in cands:
        if c["th_args"]:
            primary = c
            break
    if primary is None and cands:
        primary = cands[0]

    print("\n待补跑的 4 条命令草稿（**仅草稿，跑之前请与我核对**）：")
    if primary is None:
        print("    未找到任何带 --seed 的入口脚本，无法生成草稿。")
        print("    请手动执行：`.venv/bin/python <训练脚本> --help` 后把输出贴回。")
        return

    script = rel(root, primary["path"])
    seed_flag = primary["seed_args"][0] if primary["seed_args"] else "--seed"
    th_flag = primary["th_args"][0] if primary["th_args"] else "--frequency_threshold"
    print(f"    入口脚本：{script}")
    print(f"    seed 参数：{seed_flag}   阈值参数：{th_flag}")
    print()
    for s in SEEDS_TODO:
        cmd = (f".venv/bin/python {script} "
               f"--dataset FD002 --window_size 48 {seed_flag} {s} "
               f"{th_flag} 0.25")
        print(f"    # 种子 {s}")
        print(f"    {cmd}")
    print()
    print("    以上参数名/取值均为推测，务必先跑一次 --help 核对。")


# --------------------------------------------------------------------------
# θ 校验
# --------------------------------------------------------------------------

VERIFY_SNIPPET = r'''
import torch, glob, os, sys
pats = sys.argv[1]
rows = []
for p in sorted(glob.glob(pats)):
    try:
        o = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"{os.path.basename(p)}  [读取失败] {e}")
        continue
    sd = o.get("state_dict", o) if isinstance(o, dict) else o
    if not isinstance(sd, dict):
        print(f"{os.path.basename(p)}  [非 dict] {type(o)}")
        continue
    hits = {k: v.flatten().tolist() for k, v in sd.items()
            if "threshold_param" in k}
    rows.append((os.path.basename(p), hits))
ok = True
print()
print(f"{'checkpoint':<52} {'ups.0.2 (θ1)':>14} {'ups.1.2 (θ2)':>14}  状态")
print("-" * 100)
for name, hits in rows:
    t1 = hits.get("ups.0.2.fn.fn.threshold_param", [None])[0]
    t2 = hits.get("ups.1.2.fn.fn.threshold_param", [None])[0]
    if t1 is None and t2 is None:
        print(f"{name:<52} {'未命中':>14} {'未命中':>14}  ✗ 没找到 threshold_param")
        ok = False
        continue
    good = (t1 is not None and abs(t1 - 0.25) < 1e-6 and
            t2 is not None and abs(t2 - 0.25) < 1e-6)
    ok = ok and good
    s1 = f"{t1:.6f}" if isinstance(t1, float) else str(t1)
    s2 = f"{t2:.6f}" if isinstance(t2, float) else str(t2)
    print(f"{name:<52} {s1:>14} {s2:>14}  {'✓ 已固定' if good else '✗ 不是 0.25！'}")
print("-" * 100)
print("总结：" + ("全部通过，固定臂真实有效。" if ok
                  else "存在未固定的 checkpoint，不要把它们混进固定臂统计。"))
'''


def run_verify(root):
    py = sys.executable
    pattern = str(Path(root) / "checkpoints" / "FD002_w48_seed*_freq_threshold_025.pth")
    print(f"解释器：{py}")
    print(f"匹配：{pattern}")
    try:
        import importlib
        importlib.import_module("torch")
    except ImportError:
        print("\n当前解释器没有 torch。请改用项目虚拟环境重跑：")
        print("    .venv/bin/python backfill_fixed_arm.py --verify")
        return 1
    return subprocess.call([py, "-c", VERIFY_SNIPPET, pattern])


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="固定臂补跑：配方发现（--discover）与 θ 校验（--verify）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--discover", action="store_true", help="扫描并反推固定臂配方（默认）")
    ap.add_argument("--verify", action="store_true", help="校验已有固定臂 checkpoint 的 θ 是否为 0.25")
    ap.add_argument("--root", default=".", help="项目根目录，默认当前目录")
    ap.add_argument("--max-hits", type=int, default=25, help="每段最多打印的命中数")
    a = ap.parse_args()

    root = Path(a.root).resolve()
    if not root.is_dir():
        print(f"根目录不存在：{root}")
        return 1
    print(f"项目根目录：{root}")

    if a.verify:
        hr("【V】固定臂 θ 校验")
        return run_verify(root)

    # ---- T1: 强关键字 ----
    hr("【T1】强关键字命中（freq_threshold_025 等字面量 → 大概率就是当年的入口）")
    hits, nfiles = grep(root, KEY_STRONG, a.max_hits)
    print(f"（已扫描 {nfiles} 个文本文件，跳过 checkpoints/ 与 outputs/）")
    if hits:
        cur = None
        for f, ln, line, kw in hits:
            if f != cur:
                print(f"\n  {rel(root, f)}")
                cur = f
            print(f"    {ln:>5}: {line}")
    else:
        print("  未命中。见文末判定规则第 3 条。")

    # ---- T2: 中关键字（限缩到阈值相关，避免刷屏）----
    hr("【T2】阈值相关参数命中（frequency_threshold_init / threshold_param 等）")
    hits2, _ = grep(root, KEY_MEDIUM, a.max_hits)
    if hits2:
        cur = None
        for f, ln, line, kw in hits2:
            if f != cur:
                print(f"\n  {rel(root, f)}")
                cur = f
            print(f"    {ln:>5}: {line}")
    else:
        print("  未命中。")

    # ---- T3: 候选入口 ----
    hr("【T3】候选训练入口（含 argparse --seed 的 .py）")
    cands = scan_entrypoints(root)
    if not cands:
        print("  未找到带 --seed 的 argparse 脚本。")
    for c in cands[:a.max_hits]:
        mark = "★" if c["th_args"] else " "
        print(f"\n {mark} {rel(root, c['path'])}   (共 {c['n_args']} 个参数)")
        print(f"     seed   : {', '.join(c['seed_args']) or '—'}")
        print(f"     threshold: {', '.join(c['th_args']) or '—（该脚本无阈值参数！）'}")
        print(f"     命名类 : {', '.join(c['flag_args']) or '—'}")

    # ---- T4: 已有产物 ----
    hr("【T4】固定臂已有产物")
    dirs, files = dump_existing(root, a.max_hits)
    print(f"相关目录 {len(dirs)} 个：")
    for d in dirs:
        print(f"  {rel(root, d)}")
    print(f"\n相关文件（前 {a.max_hits} 个）：")
    for f in files:
        sz = Path(f).stat().st_size
        print(f"  {rel(root, f)}   ({sz} B)")

    print("\n其中 json 元数据的内容（若存在，超参以这里为准）：")
    shown = 0
    for f in files:
        if Path(f).suffix.lower() != ".json":
            continue
        if shown >= 4:
            print("  ...（其余略，见【T4】清单）")
            break
        print(f"\n  ── {rel(root, f)}")
        show_json(f)
        shown += 1
    if shown == 0:
        print("  没有可读的 json 元数据，超参只能靠脚本内的默认值 + 命令行推断。")

    # ---- T5: 计划 ----
    hr("【T5】补跑计划")
    print(f"已完成种子：{SEEDS_DONE}（n=3，不足以做统计）")
    print(f"待补种子  ：{SEEDS_TODO}")
    print(f"目标      ：7 对 7 逐种子配对，配对 t 检验")
    print("配置必须与已跑的固定臂逐字一致：")
    print("  Adam / lr 0.002 / warm-up+cosine / 70 epoch / batch 256 / grad_clip 1.0")
    print("  DDPM / T=1000 / linear 噪声计划")
    print("评估：切分种子 20260915，预测器 70/30，判别器 80/20，")
    print("      评估器种子 3/13/23/33/43（每份 5 次）→ 4 × 5 = 20 次评估")
    print("预计：训练 ~200 s/种子 + 采样 ~1850 s/种子 ≈ 35 min/种子，4 个约 2.4 h")
    emit_plan(root, cands, hits)

    hr("【请把输出整段贴回】")
    print("我把 4 条命令按你实际的脚本签名定稿后再跑，避免参数名猜错白烧 4 小时。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
