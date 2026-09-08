# Capture Log

## Phase 0

### Camera rendering benchmark

- Renderer: PyBullet TinyRenderer
- Resolution: 128x128
- Frames: 200
- Render FPS: 48.35
- Environment: WSL2 Ubuntu
- Output image: `results/camera_test.png`

TinyRenderer was kept as the default because the hardware OpenGL path showed no measurable acceleration in this environment.


### Closed-loop vision transfer

Using the same frozen PPO policy with camera-estimated goal positions:

- Privileged PPO success: 100%
- Single vision estimate at reset: 91%
- Vision estimate every control step: 27%

Perception-in-the-loop:
- Mean 3D pose error: 41.58 cm
- Median 3D pose error: 42.99 cm
- P90 error: 50.57 cm
- P95 error: 53.03 cm
- Mean final distance: 34.05 cm

The large closed-loop degradation exposed a distribution shift: the pose estimator was trained on reset-state images but encountered moving robot configurations during deployment.

### Perception distribution shift

Pose error was measured while the privileged PPO controlled the robot:

- Step 0: 2.32 cm mean error
- Step 1: 22.13 cm mean error
- Step 2: 39.90 cm mean error
- Step 3: 45.09 cm mean error

Because the controller still used privileged state, this isolates robot-motion-induced visual distribution shift from feedback errors caused by the vision policy itself.

### Closed-loop recovery with control-state training data

The reset-only estimator failed under robot motion:

- Naive closed-loop success: 27%
- Mean in-loop pose error: 41.58 cm

After retraining the same ResNet-18 architecture on images collected from privileged PPO trajectories:

- Control-state test mean error: 2.32 cm
- Control-state test P95 error: 4.90 cm
- Closed-loop success: 95%
- Mean steps: 5.29
- Mean final distance: 3.73 cm
- Mean episode pose error: 2.97 cm

This recovered closed-loop performance without changing the network architecture, showing that the dominant failure mode was dataset distribution shift rather than insufficient model capacity.