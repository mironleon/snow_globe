from pathlib import Path

import numpy as np
import numpy.typing as npt
from plotly import graph_objects as go

from snow_globe.geometry import rotate_array
from snow_globe.surface import normal_perturbation


class SnowFlake:
    def __init__(
        self,
        n_segments: int = 25,
        segment_length: int = 20,
        segment_length_sigma: float = 15,
        angle_sigma: float = np.pi / 1.5,
        deterministic: bool = False,
    ):
        self.n_segments = n_segments
        self.segment_length = segment_length
        self.segment_length_sigma = segment_length_sigma
        self.angle_sigma = angle_sigma
        self.deterministic = deterministic
        self.pos = self._generate_pos()

    def _generate_pos(self):
        """
        Generate ordered positions which will represent a hexagonally symmetric 2D
        snowflake when plotted as a line graph. Works by generating a single random line
        in the positive X and Y quadrant, copying it and flipping the y values, and
        copying and rotating clockwise by 60 degrees 5 times.
        """
        if self.deterministic:
            np.random.seed(10)
        # Total will include the origin twice, and the rightmost point once, thus +3
        N = 2 * self.n_segments + 3
        pos = np.zeros((N, 2))
        for i in range(self.n_segments):
            pos[i + 1] = self._get_next_pos(pos[i])
        # include rightmost points
        pos[self.n_segments + 1] = [pos[self.n_segments][0] + self.segment_length, 0]
        # copy, flip and add to original
        reverse_negative = np.copy(pos[self.n_segments :: -1])
        reverse_negative[:, 1] = -reverse_negative[:, 1]
        pos[self.n_segments + 2 :] = reverse_negative
        # make 5 copies, each rotated 60 degrees
        snowflake_pos = np.zeros((len(pos) * 6, 2), dtype=np.float64)
        snowflake_pos[:N] = pos
        for i in range(1, 6):
            angle = i * np.pi / 3
            snowflake_pos[i * N : (i + 1) * N] = rotate_array(pos, angle)
        return snowflake_pos

    def _get_next_pos(self, curr_pos: npt.NDArray[np.floating]):
        angle = normal_perturbation(0, np.pi / 1.5, bounds=(-np.pi / 2, np.pi / 2))
        seg_length = normal_perturbation(self.segment_length, self.segment_length_sigma)
        next_pos = curr_pos + seg_length * np.array([np.cos(angle), np.sin(angle)])
        if next_pos[1] <= 0.0:
            return self._get_next_pos(curr_pos)
        else:
            return next_pos

    def write_image(self, fn: str | Path, force: bool = False):
        fn = Path(fn)
        assert not fn.exists() or force
        fig = go.FigureWidget(
            [
                go.Scatter(x=self.pos[:, 0], y=self.pos[:, 1]),
            ],
            layout={
                "width": 800,
                "height": 800,
                "margin": {k: 0 for k in "btlr"},
                "paper_bgcolor": "#FFF",
                "plot_bgcolor": "#FFF",
            },
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.write_image(fn)

    def to_txt(self, fn: str | Path, force: bool = False):
        fn = Path(fn)
        assert not fn.exists() or force
        np.savetxt(fn, self.pos)

    def write_mesh(self, fn):
        ...
