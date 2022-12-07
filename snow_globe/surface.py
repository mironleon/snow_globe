import numpy as np
import numpy.typing as npt


def create_grid_pos(
    length: float, res: float, make_3D: bool = True
) -> npt.NDArray[np.floating]:
    """
    Create a two dimensional grid of dimensions length/res x length/res Optionally
    include 3rd dimension filled with zeros.
    """
    nsteps = round(length / res)
    shape = (nsteps**2, 3) if make_3D else (nsteps**2, 2)
    pos = np.zeros(shape)
    x = np.linspace(-length / 2.0, length / 2.0, nsteps)
    xx, yy = np.meshgrid(x, x)
    pos[:, :2] = np.array((xx.ravel(), yy.ravel())).T
    return pos


def create_gaussian_peak(
    sigma: float, n_sigma: int = 2, scale_factor: float = 1.0, res: float = 0.01
) -> npt.NDArray[np.floating]:
    """Create an array of gaussian values based on a 2D grid"""
    grid_length = 3 * n_sigma * sigma
    grid = create_grid_pos(length=grid_length, res=res * grid_length, make_3D=False)
    g = gaussian_2D(sigma=sigma, x=grid[:, 0], y=grid[:, 1])
    return scale_factor * g


def create_height_data(
    grid_size: int,
    n_peaks: tuple[int],
    sigmas: tuple[float],
    scale_factors: tuple[float],
) -> npt.NDArray[np.floating]:
    height_data = np.zeros((grid_size, grid_size))
    peak_res = 0.1
    peak_grid_size = round(1 / peak_res)
    assert grid_size > 4 * peak_grid_size
    for n, sigma_base, scale_factor_base in zip(n_peaks, sigmas, scale_factors):
        for _ in range(n):
            sigma = np.random.normal(loc=sigma_base, scale=0.1 * sigma_base)
            scale_factor = np.random.normal(
                loc=scale_factor_base, scale=0.1 * scale_factor_base
            )
            g = create_gaussian_peak(
                sigma=sigma, scale_factor=scale_factor, res=peak_res
            )
            i, j = np.random.randint(0, grid_size - peak_grid_size, size=2)
            height_data[i : i + peak_grid_size, j : j + peak_grid_size] += g.reshape(
                (peak_grid_size, peak_grid_size)
            )
    return height_data


def gaussian_2D(
    sigma: float, x: npt.NDArray[np.floating], y: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """2D normalized gaussian distribution: (1 / 2πσ^2) * e^-((x^2 + y^2) / 2σ^2)"""
    return (
        1.0
        / (2 * np.pi * sigma**2)
        * np.exp(-(np.power(x, 2) + np.power(y, 2)) / (2 * sigma**2))
    )


def triangulate_regular_grid(grid_size: int) -> npt.NDArray[np.uint32]:
    """
    Create array of indices to triangulate a regular grid mesh. Example for grid of 4
    points with indices 0, 1, 2, 3

            2                       3
            xxxxxxxxxxxxxxxxxxxxxxxxx
            xxxx                    x
            x  xx                   x
            x    xx                 x
            x      xx               x
            x       xxx             x
            x          xxx          x
            x             xx        x
            x               xx      x
            x                xx     x
            x                 xxx   x
            x                   xx  x
            x                     xxx
            xxxxxxxxxxxxxxxxxxxxxxxxx
            0                       1

    Would produce indices for two triangles:
        [
            [0, 1, 2],
            [1, 3, 2]
        ]
    """
    row_indices = np.empty(shape=(2 * (grid_size - 1), 3), dtype=np.uint32)
    for i in range(0, grid_size - 1):
        idx = 2 * i
        row_indices[idx] = [i, i + 1, i + grid_size]
        row_indices[idx + 1] = [i + 1, i + grid_size + 1, i + grid_size]
    indices = np.zeros((2 * (grid_size - 1) ** 2, 3))
    for i in range(0, grid_size - 1):
        indices[i * (grid_size - 1) * 2 : (i + 1) * (grid_size - 1) * 2] = (
            row_indices + i * grid_size
        )
    # TODO make sure astype is not required
    return indices.astype(np.uint32)


def generate_mountain_mesh(
    grid_length: float = 512, grid_res: float = 1.0
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]:
    # TODO test
    grid_size = round(grid_length / grid_res)
    grid_pos = create_grid_pos(grid_length, grid_res)
    z = create_height_data(
        grid_size=grid_size, n_peaks=(80,), sigmas=(1.0,), scale_factors=(1000.0,)
    )
    grid_pos[:, 2] = z.ravel()
    indices = triangulate_regular_grid(grid_size)
    # TODO maybe return a trimesh object
    return grid_pos, indices
