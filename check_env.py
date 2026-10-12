"""环境自检脚本 — HybridSN 复现用

用法（在项目目录下）:
    python check_env.py

输出全部为 ASCII，避免 Windows 控制台编码问题。
可作为《环境说明》的附件证据。
"""
import platform
import sys

line = "=" * 62
print(line)
print("HybridSN reproduction - environment self check")
print(line)

# ---------- Python ----------
print("\n[Python]")
print("  version    :", sys.version.split()[0])
print("  executable :", sys.executable)
print("  platform   :", platform.platform())

ok_all = True


def check(label, cond, detail=""):
    global ok_all
    tag = "OK  " if cond else "FAIL"
    if not cond:
        ok_all = False
    print(f"  [{tag}] {label}{(' -> ' + detail) if detail else ''}")


# ---------- 核心依赖 ----------
print("\n[Core packages]")
try:
    import torch
    print("  torch      :", torch.__version__)
except Exception as e:
    check("import torch", False, str(e))
    sys.exit(1)

for mod, label in [("torchvision", "torchvision"), ("torchinfo", "torchinfo"),
                   ("numpy", "numpy"), ("scipy", "scipy"),
                   ("sklearn", "scikit-learn"), ("matplotlib", "matplotlib"),
                   ("spectral", "spectral"), ("tqdm", "tqdm")]:
    try:
        m = __import__(mod)
        print(f"  {label:<12}:", getattr(m, "__version__", "?"))
    except Exception as e:
        check(f"import {label}", False, str(e))

# ---------- CUDA / GPU ----------
print("\n[CUDA / GPU]")
print("  cuda build :", torch.version.cuda)
print("  cudnn      :", torch.backends.cudnn.version())
avail = torch.cuda.is_available()
check("torch.cuda.is_available()", avail, str(avail))
if avail:
    print("  gpu name   :", torch.cuda.get_device_name(0))
    print("  capability :", torch.cuda.get_device_capability(0))
    print("  gpu memory : %.2f GB" % (torch.cuda.get_device_properties(0).total_memory / 1024 ** 3))
    # GPU 冒烟测试：真正跑一次矩阵乘，证明 GPU 可用
    import time
    x = torch.randn(1500, 1500, device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        y = x @ x
    torch.cuda.synchronize()
    dt = time.time() - t0
    check("GPU matmul smoke test", float(y[0, 0]) == float(y[0, 0]), f"{dt:.3f}s for 10x 1500^3")
else:
    print("  !! GPU not available - training would run on CPU (very slow)")

# ---------- 数据文件 ----------
print("\n[Dataset files]")
import os
here = os.path.dirname(os.path.abspath(__file__))
need = [
    r"dataset\PaviaU\PaviaU.mat",
    r"dataset\PaviaU\PaviaU_gt.mat",
]
for rel in need:
    p = os.path.join(here, rel)
    exists = os.path.exists(p)
    size = ("%.1f MB" % (os.path.getsize(p) / 1024 ** 2)) if exists else "-"
    check(rel, exists, size)

# 试读一次 PaviaU，确认真的能读
try:
    from scipy.io import loadmat
    d = loadmat(os.path.join(here, r"dataset\PaviaU\PaviaU.mat"))
    g = loadmat(os.path.join(here, r"dataset\PaviaU\PaviaU_gt.mat"))
    cube = d["paviaU"]
    gt = g["paviaU_gt"]
    print("  cube shape :", cube.shape, cube.dtype)
    print("  gt shape   :", gt.shape, gt.dtype)
    print("  classes    :", len(set(gt.ravel().tolist())) - 1, "(excluding background 0)")
    print("  labeled px :", int((gt != 0).sum()))
    check("PaviaU readable", cube.shape == (610, 340, 103) and gt.shape == (610, 340))
except Exception as e:
    check("read PaviaU", False, str(e))

# ---------- 结论 ----------
print("\n" + line)
print("RESULT:", "ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED - see [FAIL] above")
print(line)
sys.exit(0 if ok_all else 1)
