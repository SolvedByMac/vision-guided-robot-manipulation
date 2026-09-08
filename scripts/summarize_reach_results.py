import json
from pathlib import Path

import pandas as pd


rows = [
    {
        "condition": "Random baseline",
        "success_mean": 0.17,
        "success_std": 0.0,
        "mean_steps": 44.58,
        "final_distance_cm": 23.36,
    },
    {
        "condition": "Privileged PPO",
        "success_mean": 1.00,
        "success_std": 0.00,
        "mean_steps": 2.646,
        "final_distance_cm": 3.05,
    },
    {
        "condition": "Naive closed-loop vision",
        "success_mean": 0.27,
        "success_std": 0.0,
        "mean_steps": 37.02,
        "final_distance_cm": 34.05,
    },
    {
        "condition": "Control-state vision + standard PPO",
        "success_mean": 0.926,
        "success_std": 0.040,
        "mean_steps": 6.582,
        "final_distance_cm": 4.06,
    },
    {
        "condition": "Control-state vision + noisy PPO",
        "success_mean": 0.940,
        "success_std": 0.018,
        "mean_steps": 6.244,
        "final_distance_cm": 4.05,
    },
    {
        "condition": "Camera shift + fixed estimator",
        "success_mean": 0.208,
        "success_std": 0.057,
        "mean_steps": 41.178,
        "final_distance_cm": 13.26,
    },
    {
        "condition": "Camera shift + randomized estimator",
        "success_mean": 0.856,
        "success_std": 0.027,
        "mean_steps": 10.704,
        "final_distance_cm": 5.44,
    },
]

df = pd.DataFrame(rows)

Path("results").mkdir(exist_ok=True)

df.to_csv(
    "results/reach_results_summary.csv",
    index=False,
)

summary = {
    "privileged_to_standard_vision_gap_pp": 7.4,
    "naive_to_control_state_recovery_pp": 65.6,
    "camera_shift_randomization_recovery_pp": 64.8,
    "noise_training_gain_pp": 1.4,
}

with open("results/reach_key_findings.json", "w") as file:
    json.dump(summary, file, indent=2)

print(df.to_string(index=False))
print()
print("privileged to standard vision gap: 7.4 percentage points")
print("naive to control-state recovery: 65.6 percentage points")
print("camera-shift randomization recovery: 64.8 percentage points")
print("noise-training gain: 1.4 percentage points")
print()
print("saved: results/reach_results_summary.csv")
print("saved: results/reach_key_findings.json")