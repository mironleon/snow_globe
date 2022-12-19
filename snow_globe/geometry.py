from typing import NewType

import numpy as np
import numpy.typing as npt

AngleRadians = NewType("AngleRadians", float)


def rotate_array(arr: npt.ArrayLike, angle: AngleRadians) -> npt.NDArray[np.floating]:
    arr = np.array(arr, dtype=np.float64)
    rot_matrix = get_clockwise_rotation_matrix_2D(angle)
    # have to invert rotation matrix to preserve format of position array
    return np.linalg.inv(rot_matrix).dot(arr)


def get_clockwise_rotation_matrix_2D(theta: AngleRadians) -> npt.NDArray[np.floating]:
    return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
