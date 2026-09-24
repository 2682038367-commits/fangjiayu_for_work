#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从已有 checkpoint 里读出频率阈值 θ（不需要重训，不需要 GPU）

背景：论文声称 θ 是可学习参数，公开代码里它随机初始化后被梯度链切断而冻结。
      manifest / metrics 里没有记录 θ，但它作为 nn.Parameter 一定在 state_dict 里。

优先用 torch 读；若当前解释器没有 torch（例如用了系统 python3 而不是项目 venv），
自动降级到内置的纯 numpy 读取器，效果一样。

用法：
    # 第一步：不知道权重在哪，先定位（列出候选文件 + 从 json/yaml 里挖路径）
    .venv/bin/python read_theta_from_ckpt.py --locate --scan .

    # 第二步：读 θ
    .venv/bin/python read_theta_from_ckpt.py --scan results/runs --csv theta_values.csv
    .venv/bin/python read_theta_from_ckpt.py --file path/to/model.pt
    .venv/bin/python read_theta_from_ckpt.py --file path/to/model.pt --all-params

输出：每个 checkpoint 里所有名字含 threshold/theta 的参数及数值。
      典型情况会看到两个值（网络里有两个频率模块，各一个 θ）。
"""

import argparse
import csv
import json
import os
import pickle
import re
import struct
import sys
import zipfile

try:
    import torch
except ImportError:
    torch = None

try:
    import numpy as np
except ImportError:
    np = None

CKPT_SUFFIXES = ("*.pt", "*.pth", "*.ckpt", "*.bin", "*.tar",
                 "*.safetensors", "*.pkl", "*.weights")
STATE_KEYS = ("state_dict", "model_state_dict", "model", "net",
              "ema_state_dict", "ema", "module")
KEY_HINTS = ("threshold", "theta", "freq_th", "quantile", "th_param")
MAX_LEN = 8          # θ 是标量或极小向量，超过这个长度多半是别的参数
SKIP_DIRS = {".venv", "venv", "env", ".git", "node_modules", "__pycache__",
             ".mypy_cache", ".pytest_cache", ".idea", ".vscode", "site-packages"}
# 从 metadata / config 里挖权重路径时用的线索
PATH_RE = re.compile(r"[\w./\\-]+\.(?:pt|pth|ckpt|bin|safetensors|tar)\b", re.I)
TEXT_SUFFIXES = (".json", ".yaml", ".yml", ".toml", ".log", ".txt", ".md", ".sh")


# --------------------------------------------------------------------------
# 无 torch 时的降级读取器：直接解 zip + pickle，把 torch 的重建函数换成 stub
# --------------------------------------------------------------------------

def _stub_rebuild(storage, storage_offset, size, stride=None,
                  requires_grad=None, backward_hooks=None):
    arr = storage
    try:
        n = int(np.prod(size))
        arr = arr.ravel()[storage_offset:storage_offset + n].reshape(size)
    except Exception:
        pass
    return arr


class _Dummy:
    """兜住 pickle 里所有 torch 名字，缺哪个补哪个。"""
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return None


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch._utils" and name.startswith("_rebuild_tensor"):
            return _stub_rebuild
        if module.startswith("torch"):
            return _Dummy
        return super().find_class(module, name)


def _dtype_from_storage_type(storage_type):
    s = (storage_type or "").lower()
    if "double" in s:
        return "float64"
    if "half" in s:
        return "float16"
    if "long" in s:
        return "int64"
    if "int" in s:
        return "int32"
    return "float32"


def _load_no_torch(path):
    """返回一个扁平 dict：参数名 -> numpy 数组（读不出来就返回 None）。"""
    if np is None:
        return None

    # ---- 新版 torch 的 zip 格式 ----
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            pkl_name = next((n for n in names if n.endswith("data.pkl")), None)
            if pkl_name is None:
                return None
            prefix = pkl_name[: -len("data.pkl")]

            def persistent_load(saved_id):
                if not isinstance(saved_id, tuple) or saved_id[0] != "storage":
                    raise pickle.UnpicklingError(f"未知 persistent id: {saved_id}")
                storage_type = str(saved_id[1])
                key = str(saved_id[2])
                cand = [n for n in names if n.endswith("/" + key) or n == key]
                if not cand:
                    raise pickle.UnpicklingError(f"找不到 storage {key}")
                raw = zf.read(cand[0])
                return np.frombuffer(raw, dtype=_dtype_from_storage_type(storage_type))

            import io
            try:
                up = _Unpickler(io.BytesIO(zf.read(pkl_name)))   # 只读自己项目的 ckpt
                up.persistent_load = persistent_load
                obj = up.load()
            except Exception as e:
                print(f"    [降级失败] {type(e).__name__}: {e}")
                return None
        return obj

    # ---- 旧版 torch：裸 pickle ----
    try:
        up = _Unpickler(open(path, "rb"))
        up.persistent_load = _Dummy
        return up.load()
    except Exception as e:
        print(f"    [降级失败] {type(e).__name__}: {e}")
        return None


_ST_DTYPES = {"F64": "float64", "F32": "float32", "F16": "float16",
              "I64": "int64", "I32": "int32", "U8": "uint8"}


def _load_safetensors(path):
    """纯 python 解析 safetensors，返回 dict: name -> numpy 数组。"""
    if np is None:
        return None
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode("utf-8"))
        blob_off = 8 + n
        out = {}
        with open(path, "rb") as g:
            for k, meta in header.items():
                if k == "__metadata__":
                    continue
                dt = meta.get("dtype")
                if dt not in _ST_DTYPES:
                    continue
                s, e = meta["data_offsets"]
                g.seek(blob_off + s)
                raw = g.read(e - s)
                out[k] = np.frombuffer(raw, dtype=_ST_DTYPES[dt])
        return out


# --------------------------------------------------------------------------
# 定位：列出候选权重文件，并从文本配置里挖出记录的权重路径
# --------------------------------------------------------------------------

def walk_files(roots):
    seen = set()
    for root in roots:
        if os.path.isfile(root):
            yield root
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                if p not in seen:
                    seen.add(p)
                    yield p


def is_ckpt(path):
    return any(path.lower().endswith(s.lstrip("*")) for s in CKPT_SUFFIXES)


def find_ckpts(roots):
    return sorted(p for p in walk_files(roots) if is_ckpt(p))


def locate(roots, topn=30):
    ckpts = find_ckpts(roots)
    print("=" * 70)
    print(f"一、扫描到的权重文件：{len(ckpts)} 个")
    print("=" * 70)
    if ckpts:
        sized = []
        for p in ckpts:
            try:
                sized.append((os.path.getsize(p), p))
            except OSError:
                sized.append((0, p))
        sized.sort(reverse=True)
        for sz, p in sized[:topn]:
            print(f"  {sz/1e6:>9.2f} MB   {p}")
        if len(sized) > topn:
            print(f"  ... 还有 {len(sized)-topn} 个未列出")
    else:
        print("  一个都没找到。")

    print()
    print("=" * 70)
    print("二、从 metadata / config / 日志里挖出的权重路径线索")
    print("=" * 70)
    hits = {}
    for p in walk_files(roots):
        if not p.lower().endswith(TEXT_SUFFIXES):
            continue
        try:
            if os.path.getsize(p) > 4_000_000:
                continue
            txt = open(p, "r", encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for m in PATH_RE.findall(txt):
            hits.setdefault(m, set()).add(os.path.basename(p))
    if hits:
        for path, srcs in sorted(hits.items()):
            print(f"  {path}")
            print(f"      出现于: {', '.join(sorted(srcs)[:4])}")
    else:
        print("  没有在文本文件里发现 .pt/.ckpt 之类的路径。")

    print()
    print("=" * 70)
    print("三、体量最大的 15 个文件（权重若用了别的后缀，多半在这里）")
    print("=" * 70)
    big = []
    for p in walk_files(roots):
        if is_ckpt(p):
            continue
        try:
            sz = os.path.getsize(p)
        except OSError:
            continue
        if sz > 500_000:
            big.append((sz, p))
    big.sort(reverse=True)
    for sz, p in big[:15]:
        print(f"  {sz/1e6:>9.2f} MB   {p}")
    if not big:
        print("  没有超过 0.5 MB 的非权重文件。")


# --------------------------------------------------------------------------
# 通用部分
# --------------------------------------------------------------------------

def unwrap(obj):
    """从各种 checkpoint 包装里掏出真正的 state_dict。"""
    if isinstance(obj, dict):
        for k in STATE_KEYS:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        vals = list(obj.values())[:5]
        if vals and all(hasattr(v, "shape") for v in vals):
            return obj
    if hasattr(obj, "state_dict"):
        try:
            return obj.state_dict()
        except Exception:
            return None
    return None


def to_list(t):
    if np is not None and isinstance(t, np.ndarray):
        return t.ravel().tolist()
    if hasattr(t, "detach"):
        t = t.detach().cpu()
    if hasattr(t, "flatten"):
        return t.flatten().tolist()
    return [float(t)]


def extract_theta(sd):
    found = []
    if not isinstance(sd, dict):
        return found
    for name, tensor in sd.items():
        low = str(name).lower()
        if not any(h in low for h in KEY_HINTS):
            continue
        if not hasattr(tensor, "shape") and not hasattr(tensor, "tolist"):
            continue
        try:
            flat = to_list(tensor)
        except Exception:
            continue
        if not flat or len(flat) > MAX_LEN:
            continue
        found.append((name, [round(float(v), 6) for v in flat]))
    return found


def load_any(path):
    """返回 (对象, 用的方式)。"""
    if path.lower().endswith(".safetensors"):
        return _load_safetensors(path), "safetensors"
    if torch is not None:
        try:
            return torch.load(path, map_location="cpu", weights_only=False), "torch"
        except Exception as e:
            print(f"    [torch 加载失败，改降级] {type(e).__name__}: {e}")
    return _load_no_torch(path), "降级"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="append", default=[],
                    help="checkpoint 根目录，递归扫描；可传多次")
    ap.add_argument("--file", action="append", default=[],
                    help="单个 checkpoint 文件；可传多次")
    ap.add_argument("--csv", help="可选：把结果导出为 CSV")
    ap.add_argument("--locate", action="store_true",
                    help="只定位不读取：列出权重候选、配置文件里的路径线索、大文件")
    ap.add_argument("--all-params", action="store_true",
                    help="顺带列出全部参数名（用于确认 θ 的真实 key 名）")
    args = ap.parse_args()

    if torch is None:
        print("提示：当前解释器没有 torch，将用内置降级读取器（纯 numpy）。")
        print("      若想走 torch 路径，用项目的 venv："
              "  .venv/bin/python read_theta_from_ckpt.py ...\n")
    if np is None:
        print("警告：连 numpy 都没有，降级路径不可用，只能靠 torch。")

    roots = args.scan or ["."]

    if args.locate:
        locate(roots)
        print("\n下一步：把上面找到的真实目录填进 --scan，"
              "或直接 --file 指定单个权重文件，即可读出 θ。")
        return

    files = list(args.file)
    for r in args.scan:
        files.extend(find_ckpts([r]))
    files = sorted(set(files))

    if not files:
        print(f"在 {', '.join(args.scan) or '(未指定目录)'} 下没找到权重文件。")
        print("先跑定位模式看看权重到底在哪：\n"
              "    .venv/bin/python read_theta_from_ckpt.py --locate --scan .\n")
        locate(roots)
        return

    rows = []
    for path in files:
        obj, mode = load_any(path)
        if obj is None:
            print(f"[跳过] {os.path.basename(path)}  读不出内容（{mode}）")
            continue
        sd = unwrap(obj)
        if sd is None:
            print(f"[跳过] {os.path.basename(path)}  没找到 state_dict"
                  f"（保存的可能是完整模型对象；用带 torch 的 venv 再试）")
            continue
        hits = extract_theta(sd)
        tag = os.path.basename(path)
        if hits:
            for name, vals in hits:
                print(f"{tag}  [{mode}]\n    {name}  θ = {vals}")
                rows.append(dict(checkpoint=path, param=name,
                                 theta_1=vals[0],
                                 theta_2=vals[1] if len(vals) > 1 else "",
                                 all_values=";".join(str(v) for v in vals)))
        else:
            print(f"{tag}  [{mode}]\n    [未命中] 没有名字含 threshold/theta 的参数")
            if args.all_params:
                for k in list(sd.keys())[:40]:
                    print(f"       {k}")

    if args.csv and rows:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["checkpoint", "param", "theta_1",
                                              "theta_2", "all_values"])
            w.writeheader()
            w.writerows(rows)
        print(f"\n已导出 {len(rows)} 条 → {args.csv}")

    print(f"\n扫描 {len(files)} 个 checkpoint，命中 {len(rows)} 条 θ 记录。")
    if not rows:
        print("若全部未命中：θ 的 key 名可能不含上述关键字，"
              "加 --all-params 看真实参数名。")


if __name__ == "__main__":
    main()
