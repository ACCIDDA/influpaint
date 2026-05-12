#!/usr/bin/env python3
"""
Read every out_rsv_*.out training log in the current directory, extract the
per-epoch "Avg Loss" lines, and plot one curve per run.

Usage (from repo root):
    python RSV/workflow/plot_losses.py
"""

import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt

PATTERN = re.compile(r"Epoch (\d+) completed - Avg Loss: ([\d.eE+-]+)")

fig, ax = plt.subplots(figsize=(10, 6))

for path in sorted(glob.glob("out_rsv_*.out")):
    losses = []
    for line in Path(path).read_text().splitlines():
        m = PATTERN.search(line)
        if m:
            losses.append(float(m.group(2)))
    if losses:
        label = path.replace("out_rsv_", "").replace(".out", "")
        ax.plot(losses, label=label, linewidth=1)

ax.set_xlabel("Epoch")
ax.set_ylabel("Avg loss")
ax.set_yscale("log")
ax.legend(fontsize=8, loc="best")
ax.set_title("RSV training loss curves")
fig.tight_layout()
fig.savefig("rsv_training_losses.png", dpi=150)
print("Saved rsv_training_losses.png")
