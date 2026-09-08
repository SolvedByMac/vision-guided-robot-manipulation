import time

import cv2

from src.envs.camera import render_camera
from src.envs.reach_env import make_reach_env


env = make_reach_env()
env.reset(seed=0)

for _ in range(10):
    render_camera()

frames = 200
start = time.perf_counter()

image = None

for _ in range(frames):
    image, _, _ = render_camera()

elapsed = time.perf_counter() - start
fps = frames / elapsed

rgb = image[:, :, :3]
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

cv2.imwrite("results/camera_test.png", bgr)

print("renderer: TinyRenderer")
print("image shape:", image.shape)
print("rendered frames:", frames)
print("elapsed seconds:", round(elapsed, 3))
print("render FPS:", round(fps, 2))
print("saved: results/camera_test.png")

env.close()