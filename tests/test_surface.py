import numpy as np
import pytest

import snow_globe.surface as surface


def test_create_grid_pos_3D():
    grid_pos = surface.create_grid_pos(length=1.5, res=0.1, make_3D=True)
    np.testing.assert_allclose(np.max(grid_pos, axis=0), np.array([0.75, 0.75, 0.0]))
    np.testing.assert_allclose(np.min(grid_pos, axis=0), np.array([-0.75, -0.75, 0.0]))
    assert grid_pos.shape == (15**2, 3)


def test_create_grid_pos_2D():
    grid_pos = surface.create_grid_pos(length=1.5, res=0.1, make_3D=False)
    np.testing.assert_allclose(np.max(grid_pos, axis=0), np.array([0.75, 0.75]))
    np.testing.assert_allclose(np.min(grid_pos, axis=0), np.array([-0.75, -0.75]))
    assert grid_pos.shape == (15**2, 2)


def test_gaussian_2d():
    res = 0.01
    sigma = 0.5
    grid_pos = surface.create_grid_pos(length=4.0, res=res, make_3D=False)
    g = surface.gaussian_2D(sigma=sigma, x=grid_pos[:, 0], y=grid_pos[:, 1])
    # gaussian is normalized, so should integrate to 1
    integral = np.sum(g) * res**2
    np.testing.assert_approx_equal(integral, 1.0, significant=2)
    # max value at 0,0 should be (1 / 2πσ^2)
    max_val = 1 / (2 * np.pi * sigma**2)
    np.testing.assert_approx_equal(np.max(g), max_val, significant=2)


@pytest.mark.skip("Messed something up with grid length/sigma calculation")
def test_create_gaussian_peak():
    sigma = 0.5
    grid_length = 4.0
    n_sigma = round(0.33 * grid_length / sigma)
    scale_factor = 3.0
    g = surface.create_gaussian_peak(
        sigma=sigma, n_sigma=n_sigma, scale_factor=scale_factor
    )
    # gaussian is normalized, so should integrate to 1 times scale factpr
    integral = np.sum(g) * (0.01 * grid_length) ** 2
    np.testing.assert_approx_equal(integral, 1.0 * scale_factor, significant=2)
    # max value at 0,0 should be (1 / 2πσ^2) * scale factor
    max_val = scale_factor / (2 * np.pi * sigma**2)
    np.testing.assert_approx_equal(np.max(g), max_val, significant=2)


@pytest.mark.parametrize(
    "size, expected",
    [
        (2, np.array([[0.0, 1.0, 2.0], [1.0, 3.0, 2.0]])),
        (
            3,
            np.array(
                [
                    [0.0, 1.0, 3.0],
                    [1.0, 4.0, 3.0],
                    [1.0, 2.0, 4.0],
                    [2.0, 5.0, 4.0],
                    [3.0, 4.0, 6.0],
                    [4.0, 7.0, 6.0],
                    [4.0, 5.0, 7.0],
                    [5.0, 8.0, 7.0],
                ]
            ),
        ),
    ],
)
def test_triangulate_regular_grid(size, expected):
    np.testing.assert_allclose(surface.triangulate_regular_grid(size), expected)


def test_create_height_data():
    height_data = surface.create_height_data(
        grid_size=512, n_peaks=(5,), sigmas=(2.0,), scale_factors=(10.0,)
    )
    assert height_data.shape == (512, 512)
