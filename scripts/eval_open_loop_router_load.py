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


# # ───────────────────────── Dataset ────────────────────────────
# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# class EpisodeSampler(torch.utils.data.Sampler):
#     def __init__(self, ds: LeRobotDataset, ep: int):
#         self.ids = range(
#             ds.episode_data_index["from"][ep].item(),
#             ds.episode_data_index["to"][ep].item() + 1,
#         )
#     def __iter__(self): return iter(self.ids)
#     def __len__(self):   return len(self.ids)

# # ───────────────────────── Inference ──────────────────────────
# from openpi_client import websocket_client_policy
# from openpi_client.runtime.agents import policy_agent as _policy_agent
# from openpi_client import action_chunk_broker


# def main(cfg):
#     # 1. policy client ----------------------------------------------------------------
#     client = websocket_client_policy.WebsocketClientPolicy(cfg.policy_host, cfg.policy_port)
#     agent = _policy_agent.PolicyAgent(
#         policy=action_chunk_broker.ActionChunkBroker(client, action_horizon=10)
#     )

#     # 2. dataset ----------------------------------------------------------------------
#     ds = LeRobotDataset(cfg.repo_id, local_files_only=True)
#     sampler = EpisodeSampler(ds, cfg.episode_index)
#     loader = torch.utils.data.DataLoader(ds, batch_size=1, sampler=sampler, num_workers=0)

#     # 3. inference ----------------------------------------------------------------------
#     print(f"Inference on episode {cfg.episode_index} ...")
#     for batch in tqdm(loader):
#         # -------- ground-truth -------
#         gt_action = batch["action"][0].numpy()[:]          # shape (7,)
#         # -------- build observation ---
#         obs = {
#             "state": batch.get("observation.state", torch.zeros(13))[0].numpy(),
#             "image": batch[ds.meta.camera_keys[0]][0].numpy(),
#             "wrist_image": batch[ds.meta.camera_keys[1]][0].numpy(),
#             "prompt": cfg.prompt,
#         }
#         # -------- policy inference ----
#         try:
#             pred = agent.get_action(obs)["actions"].tolist()            # (1,7)
#         except Exception as e:
#             print("Policy error:", e)
#             pred = np.zeros_like(gt_action)

#     # 4. report -----------------------------------------------------------------------
# if cfg.output_dir:
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
plt.xlabel("Step")
plt.ylabel("Gating probability")
plt.title("Router logits (softmax) over time")
plt.legend()

# 3) 显示 & 保存
plt.show()
# if cfg.output_dir:
fig_path = out_dir / f"gating_probs_episode.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"✓ Router-prob plot saved to {fig_path}")


# # ───────────────────────── CLI ────────────────────────────────
# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--repo-id",        required=True)
#     p.add_argument("--episode-index",  type=int, required=True)
#     p.add_argument("--prompt",         required=True)
#     p.add_argument("--policy-host",    default="localhost")
#     p.add_argument("--policy-port",    type=int, default=8000)
#     p.add_argument("--output-dir",     type=Path, default=None)
#     cfg = p.parse_args(); main(cfg)