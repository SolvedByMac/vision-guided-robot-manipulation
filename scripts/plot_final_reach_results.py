from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


labels = [
    "Privileged PPO",
    "Naive vision",
    "Control-state vision",
    "Noise-trained vision",
    "Shifted camera\nfixed estimator",
    "Shifted camera\nrandomized estimator",
]

success = np.array([
    100.0,
    27.0,
    92.6,
    94.0,
    20.8,
    85.6,
])

error = np.array([
    0.0,
    0.0,
    4.0,
    1.8,
    5.7,
    2.7,
])

x = np.arange(len(labels))

Path("results").mkdir(exist_ok=True)

plt.figure(figsize=(11, 6))

plt.bar(
    x,
    success,
    yerr=error,
    capsize=5,
)

plt.xticks(
    x,
    labels,
    rotation=12,
    ha="right",
)

plt.ylabel("Success rate (%)")
plt.ylim(0, 110)
plt.title("Vision-Guided Reach: Perception-to-Control and Generalization")

for index, value in enumerate(success):
    plt.text(
        index,
        value + error[index] + 2,
        f"{value:.1f}%",
        ha="center",
        fontsize=9,
    )

plt.tight_layout()

output_path = "results/final_reach_results.png"

plt.savefig(
    output_path,
    dpi=250,
)

plt.close()

print(f"saved: {output_path}")