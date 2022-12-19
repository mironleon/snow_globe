from pathlib import Path

import numpy as np
import numpy.typing as npt
from plotly import graph_objects as go

from snow_globe.geometry import rotate_array
from snow_globe.surface import normal_perturbation


class SnowFlake:
    def __init__(
        self,
        n_segments: int = 250,
        segment_length: int = 20,
        segment_length_sigma: float = 15,
        angle_sigma: float = np.pi / 1.5,
    ):
        self.n_segments = n_segments
        self.segment_length = segment_length
        self.segment_length_sigma = segment_length_sigma
        self.angle_sigma = angle_sigma
        self.pos = self.generate_pos()

    def generate_pos(self):
        # 0,0 included twice, plus 1 endpoint?
        N = 2 * self.n_segments + 3
        # should be N*6 and use for all positions
        pos = np.zeros((N, 2))
        for i in range(self.n_segments):
            pos[i + 1] = self._get_next_pos(pos[i])

        pos[self.n_segments + 1] = [pos[self.n_segments][0] + self.segment_length, 0]
        backward = np.copy(pos[self.n_segments :: -1])
        backward[:, 1] = -backward[:, 1]
        pos[self.n_segments + 2 :] = backward

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

    def write_image(self, fn: str | Path, force=False):
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

    def write_mesh(self, fn):
        ...
