import numpy as np
import pytest

import snow_globe.surface as surface


def test_create_grid_pos_3D():
    grid_pos = surface.create_grid_pos(length=1.5, nsteps=15, make_3D=True)
    np.testing.assert_allclose(np.max(grid_pos, axis=0), np.array([0.75, 0.75, 0.0]))
    np.testing.assert_allclose(np.min(grid_pos, axis=0), np.array([-0.75, -0.75, 0.0]))
    assert grid_pos.shape == (15**2, 3)


def test_create_grid_pos_2D():
    grid_pos = surface.create_grid_pos(length=1.5, nsteps=15, make_3D=False)
    np.testing.assert_allclose(np.max(grid_pos, axis=0), np.array([0.75, 0.75]))
    np.testing.assert_allclose(np.min(grid_pos, axis=0), np.array([-0.75, -0.75]))
    assert grid_pos.shape == (15**2, 2)


def test_gaussian_2d():
    res = 0.01
    sigma = 0.5
    grid_pos = surface.create_grid_pos(
        length=4.0, nsteps=round(4.0 / res), make_3D=False
    )
    g = surface.gaussian_2D(sigma=sigma, x=grid_pos[:, 0], y=grid_pos[:, 1])
    # gaussian is normalized, so should integrate to 1
    integral = np.sum(g) * res**2
    np.testing.assert_approx_equal(integral, 1.0, significant=2)
    # max value at 0,0 should be (1 / 2πσ^2)
    max_val = 1 / (2 * np.pi * sigma**2)
    np.testing.assert_approx_equal(np.max(g), max_val, significant=2)


def test_create_gaussian_peak():
    sigma = 0.5
    grid_length = 6 * sigma
    height_factor = 3.0
    res = 0.01
    g = surface.create_gaussian_peak(
        sigma=sigma, n_pixels=round(1 / res), height_factor=height_factor
    )
    # gaussian is normalized, so should integrate to 1 times scale factpr
    integral = np.sum(g) * (res * grid_length) ** 2
    np.testing.assert_approx_equal(integral, 1.0 * height_factor, significant=2)
    # max value at 0,0 should be (1 / 2πσ^2) * scale factor
    max_val = height_factor / (2 * np.pi * sigma**2)
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
        grid_size=512,
        n_peaks=(5,),
        sigmas=(2.0,),
        scale_factors=(10.0,),
        peak_widths=(100,),
    )
    assert height_data.shape == (512, 512)


def test_generate_mountain_mesh():
    pos, indices = surface.generate_mountain_mesh(grid_length=512, grid_res=1.0)
    assert pos.shape == (512**2, 3)
    assert pos.dtype == float
    assert indices.shape[1] == 3
    assert indices.ndim == 2
    assert indices.dtype == np.uint32
