import numpy as np
import pytest

import snow_globe.geometry as geometry


@pytest.mark.parametrize(
    "angle, expected",
    [
        (0, np.array([[1, 0], [0, 1]])),
        (np.pi / 2, np.array([[0, 1], [-1, 0]])),
        (np.pi, np.array([[-1, 0], [0, -1]])),
        (3 * np.pi / 2, np.array([[0, -1], [1, 0]])),
        (np.pi / 6, np.array([[np.sqrt(3) / 2, 0.5], [-0.5, np.sqrt(3) / 2]])),
    ],
)
def test_get_rotation_matrix(angle, expected):
    np.testing.assert_almost_equal(
        geometry.get_clockwise_rotation_matrix_2D(angle), expected
    )


@pytest.mark.parametrize(
    "arr, angle, expected",
    [
        (np.array([[1, 0], [0, 1]]), 0, np.array([[1, 0], [0, 1]])),
        (np.array([[1, 0], [0, 1]]), np.pi / 2, np.array([[0, -1], [1, 0]])),
        (
            np.array([[1, 0], [0, 1]]),
            np.pi / 6,
            np.array([[np.sqrt(3) / 2, -0.5], [0.5, np.sqrt(3) / 2]]),
        ),
        (
            np.array([[1, 0], [0, 1], [1, 0], [0, 1]]),
            np.pi / 6,
            np.array(
                [
                    [np.sqrt(3) / 2, -0.5],
                    [0.5, np.sqrt(3) / 2],
                    [np.sqrt(3) / 2, -0.5],
                    [0.5, np.sqrt(3) / 2],
                ]
            ),
        ),
    ],
)
def test_rotate_array(arr, angle, expected):
    np.testing.assert_almost_equal(geometry.rotate_array(arr, angle), expected)
