import numpy as np
import pybullet as p


def render_camera(width=128, height=128):
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[1.2, 1.0, 0.9],
        cameraTargetPosition=[0.0, 0.0, 0.4],
        cameraUpVector=[0.0, 0.0, 1.0],
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=width / height,
        nearVal=0.1,
        farVal=3.0,
    )

    _, _, rgba, depth, segmentation = p.getCameraImage(
        width,
        height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER,
    )

    rgba = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)
    depth = np.asarray(depth).reshape(height, width)
    segmentation = np.asarray(segmentation).reshape(height, width)

    return rgba, depth, segmentation