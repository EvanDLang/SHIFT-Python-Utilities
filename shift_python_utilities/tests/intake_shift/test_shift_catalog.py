import pytest
from shift_python_utilities.intake_shift import shift_catalog

@pytest.fixture
def cat():
    return shift_catalog()


def test_clip_raster(cat):
    # Verify each type of dataset can be read in
    cat.aviris_v1_gridded().read_chunked()
    cat.L2a(date="20220228", time="183924").read_chunked()
    cat.L1.rdn(date="20220228", time="183924").read_chunked()
    cat.L1.glt(date="20220228", time="183924").read_chunked()
    cat.L1.obs(date="20220228", time="183924").read_chunked()
    cat.L1.igm(date="20220228", time="183924").read_chunked()
    