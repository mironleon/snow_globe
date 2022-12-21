import numpy as np

from snow_globe.snowflake import SnowFlake


def test_snowflake_deterministic():
    np.random.seed(10)
    pos_1 = SnowFlake().pos
    np.random.seed(10)
    pos_2 = SnowFlake().pos
    np.testing.assert_allclose(pos_1, pos_2)


def test_write_image(tmp_path):
    fn = tmp_path / "test.svg"
    SnowFlake().write_image(fn)
    assert fn.exists()


def test_to_txt(tmp_path):
    fn = tmp_path / "test.txt"
    s = SnowFlake()
    s.to_txt(fn)
    np.testing.assert_allclose(s.pos, np.loadtxt(fn))


def test_compare_to_sample(testdatadir, tmp_path):
    with open(testdatadir / "test_snowflake.png", "rb") as f:
        sample_png = f.read()
    fn = tmp_path / "test.png"
    SnowFlake(deterministic=True).write_image(fn)
    with open(fn, "rb") as f:
        png = f.read()
    assert sample_png == png
