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
