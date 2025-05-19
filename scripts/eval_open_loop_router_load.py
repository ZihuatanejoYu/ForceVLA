#!/usr/bin/env python
"""Open-loop inference on a single episode (LeRobot image-mode) and MSE report."""

import argparse, sys, time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
import torch


out_dir = Path("/home/hairuo/flexiv_pi0/results")
out_dir.mkdir(parents=True, exist_ok=True)

# 1) 读取 (n,4) 概率张量
probs_path = Path("/home/hairuo/flexiv_pi0/router_load/gating_probs_all.pt")
probs = torch.load(probs_path, map_location="cpu")          # Tensor (n, 4)
probs_np = probs.numpy()                                    # ndarray
stride = 5000                     # 每 50 个 step 采 1 个点
subsamp = probs_np[::stride]

# 2) 绘图
import matplotlib.pyplot as plt
import numpy as np

plt.figure()
n_steps = probs_np.shape[0]
for i in range(4):
    plt.plot(np.arange(len(subsamp))*stride, subsamp[:, i], label=f"Expert {i}")
plt.xlabel("Forward Step")
plt.ylabel("Router Probability")
plt.legend()

# 3) 显示 & 保存
plt.show()
# if cfg.output_dir:
fig_path = out_dir / f"gating_probs_episode.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"✓ Router-prob plot saved to {fig_path}")