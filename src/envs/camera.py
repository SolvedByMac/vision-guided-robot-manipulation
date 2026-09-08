import numpy as np
import pybullet as p


def get_camera_matrices(width=256, height=256):
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[0.8, 0.8, 0.65],
        cameraTargetPosition=[0.0, 0.0, 0.15],
        cameraUpVector=[0.0, 0.0, 1.0],
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=45,
        aspect=width / height,
        nearVal=0.1,
        farVal=2.0,
    )

    return view_matrix, projection_matrix


def render_camera(width=256, height=256):
    view_matrix, projection_matrix = get_camera_matrices(width, height)

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