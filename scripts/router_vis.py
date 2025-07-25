#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-run MoE routing visualiser for flexiv peel cucumber experiments.
• Automatically finds all runX.pt files in directory
• Processes each run by dividing into 100 windows
• Averages the routing probabilities across runs
• English labels, Times New Roman, custom colours
• PNG + PDF output
"""

import argparse
from pathlib import Path
import math
import glob

import numpy as np
import torch
import matplotlib.pyplot as plt

# -------- Global Style --------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stixsans",
})

COLORS = ["#99A4BC", "#BCCBB2", "#E38C7A", "#A77979"]   # four experts


def find_run_files(directory: str, pattern: str = "/home/hairuo/flexiv_pi0/router_load/gating_probs_flexiv_test_flexiv_wipe_board_inputForce_run*.pt") -> list:
    """Find all run files matching the pattern in the directory"""
    search_path = Path(directory) / pattern
    return sorted(glob.glob(str(search_path)), key=lambda x: int(x.split('run')[-1].split('.')[0]))


def process_run(probs: np.ndarray, num_windows: int = 100) -> np.ndarray:
    """Process a single run into averaged windows"""
    N = probs.shape[0]
    win = max(1, N // num_windows)  # Ensure at least 1 token per window
    bins = math.ceil(N / win)
    
    top1 = probs.argmax(axis=1)
    counts = np.zeros((bins, 4), dtype=int)
    
    for i in range(bins):
        seg = top1[i * win: min((i + 1) * win, N)]
        for e in range(4):
            counts[i, e] = np.count_nonzero(seg == e)
    
    ratios = counts / counts.sum(axis=1, keepdims=True)
    return ratios


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="/home/hairuo/flexiv_pi0/router_load", 
                   help="Directory containing run files")
    p.add_argument("--out", default=None, help="output prefix (no ext)")
    args = p.parse_args()

    # ---------- Find and load all run files ----------
    run_files = find_run_files(args.dir)
    if not run_files:
        raise FileNotFoundError(f"No run files found in {args.dir}")
    
    print(f"Found {len(run_files)} run files:")
    for f in run_files:
        print(f"  {Path(f).name}")
    
    all_ratios = []
    for run_file in run_files:
        probs = torch.load(run_file, map_location="cpu")
        if not (isinstance(probs, torch.Tensor) and probs.ndim == 2 and probs.shape[1] == 4):
            raise ValueError(f"Expect Tensor of shape (N,4) in {run_file}")
        probs = probs.numpy()
        
        ratios = process_run(probs)
        all_ratios.append(ratios)
    
    # Pad shorter runs with NaNs to align lengths
    max_len = max(r.shape[0] for r in all_ratios)
    padded_ratios = []
    
    for r in all_ratios:
        pad_len = max_len - r.shape[0]
        if pad_len > 0:
            padded = np.vstack([r, np.full((pad_len, 4), np.nan)])
        else:
            padded = r
        padded_ratios.append(padded)
    
    # Average across runs (ignoring NaNs)
    avg_ratios = np.nanmean(np.stack(padded_ratios), axis=0)
    bins = avg_ratios.shape[0]

    # ---------- Plot ----------
    fig, ax = plt.subplots(figsize=(8, 6))  # Slightly wider canvas

    x = np.arange(bins-1)
    bottom = np.zeros_like(x, dtype=float)
    for e in range(4):
        ax.bar(x, avg_ratios[:-1, e], bottom=bottom,
               color=COLORS[e], width=0.9,
               label=rf"\textbf{{Expert {e}}}")
        bottom += avg_ratios[:-1, e]

    ax.set_xlabel(r"\textbf{Task Progress}", fontsize=12)
    ax.set_ylabel(r"\textbf{Average Top-1 Routing Probability}", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, bins - 1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)

    ax.legend(loc="lower center",
              bbox_to_anchor=(0.5, 1.05),
              ncol=4, frameon=False, fontsize=10)

    plt.tight_layout()

    # ---------- Save ----------
    if args.out:
        prefix = Path(args.out)
    else:
        prefix = Path(args.dir) / "flexiv_wipe_board_avg_routing"
    
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {prefix}.pdf and {prefix}.png")


if __name__ == "__main__":
    main()