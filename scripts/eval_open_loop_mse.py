#!/usr/bin/env python
"""Open-loop inference on a single episode (LeRobot image-mode) and MSE report."""

import argparse, sys, time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm


# ───────────────────────── Dataset ────────────────────────────
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, ds: LeRobotDataset, ep: int):
        self.ids = range(
            ds.episode_data_index["from"][ep].item(),
            ds.episode_data_index["to"][ep].item() + 1,
        )
    def __iter__(self): return iter(self.ids)
    def __len__(self):   return len(self.ids)

# ───────────────────────── Inference ──────────────────────────
from openpi_client import websocket_client_policy
from openpi_client.runtime.agents import policy_agent as _policy_agent
from openpi_client import action_chunk_broker


def mse(pred, gt):
    se = (pred - gt) ** 2
    return np.mean(se[:3]), np.mean(se[3:6]), se[6]


def main(cfg):
    # 1. policy client ----------------------------------------------------------------
    client = websocket_client_policy.WebsocketClientPolicy(cfg.policy_host, cfg.policy_port)
    agent = _policy_agent.PolicyAgent(
        policy=action_chunk_broker.ActionChunkBroker(client, action_horizon=10)
    )

    # 2. dataset ----------------------------------------------------------------------
    ds = LeRobotDataset(cfg.repo_id, local_files_only=True)
    sampler = EpisodeSampler(ds, cfg.episode_index)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, sampler=sampler, num_workers=0)

    # 3. metrics ----------------------------------------------------------------------
    pos_mses, rot_mses, grip_mses = [], [], []

    print(f"Inference on episode {cfg.episode_index} ...")
    for batch in tqdm(loader):
        # -------- ground-truth -------
        gt_action = batch["action"][0].numpy()[:]          # shape (7,)
        # -------- build observation ---
        obs = {
            "state": batch.get("observation.state", torch.zeros(13))[0].numpy(),
            "image": batch[ds.meta.camera_keys[0]][0].numpy(),
            "wrist_image": batch[ds.meta.camera_keys[1]][0].numpy(),
            "prompt": cfg.prompt,
        }
        # -------- policy inference ----
        try:
            pred = agent.get_action(obs)["actions"].tolist()            # (1,7)
        except Exception as e:
            print("Policy error:", e)
            pred = np.zeros_like(gt_action)

        # -------- metric --------------
        p, r, g = mse(pred, gt_action)
        pos_mses.append(p); rot_mses.append(r); grip_mses.append(g)

    # 4. report -----------------------------------------------------------------------
    print("\n=== MSE summary ===")
    print(f"Position : {np.mean(pos_mses):.6f}")
    print(f"Rotation : {np.mean(rot_mses):.6f}")
    print(f"Gripper  : {np.mean(grip_mses):.6f}")

# ───────────────────────── CLI ────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id",        required=True)
    p.add_argument("--episode-index",  type=int, required=True)
    p.add_argument("--prompt",         required=True)
    p.add_argument("--policy-host",    default="localhost")
    p.add_argument("--policy-port",    type=int, default=8000)
    cfg = p.parse_args(); main(cfg)