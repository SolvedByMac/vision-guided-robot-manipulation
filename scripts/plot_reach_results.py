from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


labels = [
    "Random",
    "Privileged PPO",
    "Naive vision",
    "Control-state vision",
    "Noise-trained vision",
]

success = np.array([
    0.17,
    1.00,
    0.27,
    0.926,
    0.940,
])

error = np.array([
    0.0,
    0.0,
    0.0,
    0.040,
    0.018,
])

x = np.arange(len(labels))

Path("results").mkdir(exist_ok=True)

plt.figure(figsize=(10, 5))

plt.bar(
    x,
    success * 100,
    yerr=error * 100,
    capsize=5,
)

plt.xticks(
    x,
    labels,
    rotation=15,
    ha="right",
)

plt.ylabel("Success rate (%)")
plt.ylim(0, 110)
plt.title("Perception-to-Control Gap on Panda Reach")

for index, value in enumerate(success):
    plt.text(
        index,
        value * 100 + error[index] * 100 + 2,
        f"{value * 100:.1f}%",
        ha="center",
    )

plt.tight_layout()
plt.savefig(
    "results/reach_success_comparison.png",
    dpi=200,
)
plt.close()

print("saved: results/reach_success_comparison.png")