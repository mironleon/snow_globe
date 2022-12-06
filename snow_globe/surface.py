import numpy as np
import numpy.typing as npt


def create_grid(size: int) -> npt.NDArray[np.floating]:
    pos = np.zeros((size**2, 3))
    x = np.arange(size)
    y = np.arange(size)
    xx, yy = np.meshgrid(x, y)
    pos[:, :2] = np.array((xx.ravel(), yy.ravel())).T
    return pos


def create_height_data(size: int) -> npt.NDArray[np.floating]:
    z_matrix = np.zeros((size, size))
    for _ in range(100):
        width = int(0.4 * size * np.random.rand())
        peak_size = int(0.02 * size * np.random.rand())
        i, j = np.random.randint(0, size - width, size=2)
        z_matrix[i : i + width, j : j + width] += create_gaussian_peak(
            width=width, height=peak_size, size=width
        )
    for _ in range(200):
        width = int(0.4 * size * np.random.rand())
        peak_size = int(0.01 * size * np.random.rand())
        i, j = np.random.randint(0, size - width, size=2)
        z_matrix[i : i + width, j : j + width] += create_gaussian_peak(
            width=width, height=peak_size, size=width
        )
    return z_matrix


def create_gaussian_peak(
    height: float, width: float, size: int = 5
) -> npt.NDArray[np.float32]:
    # size preferably an odd number
    center = int(size / 2)
    grid = create_grid(size=size)[:, :2]
    d = np.linalg.norm(grid - center, axis=1)
    gaussian = height * np.exp(-(d**2) / (2 * width))
    return gaussian.reshape(size, size)


def create_mesh_indices(size: int):
    row_indices = []

    for i in range(0, size - 1, 1):
        row_indices.append([i, i + 1, i + size])
        row_indices.append([i + 1, i + size + 1, i + size])

    row_indices_arr = np.array(row_indices)
    indices = np.zeros(((size - 1) ** 2 * 2, 3))
    for i in range(0, size - 1):
        indices[i * (size - 1) * 2 : (i + 1) * (size - 1) * 2] = (
            row_indices_arr + i * size
        )
    return indices
