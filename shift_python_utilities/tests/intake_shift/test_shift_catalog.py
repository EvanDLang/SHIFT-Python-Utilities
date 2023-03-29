import pytest
import numpy as np
from shift_python_utilities.intake_shift import shift_catalog

@pytest.fixture
def cat():
    return shift_catalog()

def test_aviris_data(cat):
    cat.aviris_v1_gridded().read_chunked()

    
@pytest.fixture
def band_dict():
    return {
        'L2a': 425,
        'rdn': 425,
        'igm': 3,
        'glt': 2,
        'obs': 11
    }
@pytest.fixture
def band_dict_filtered():
    return {
        'L2a': 337,
        'rdn': 337,
        'igm': 3,
        'glt': 2,
        'obs': 11,
    }


@pytest.mark.parametrize(
    "dataset, date, time, ortho, filter_bad_bands",
    [
        (("L1", "glt"), "20220228", "183924", False, False),
        (("L1", "glt"), "20220228", "183924", False, True),
        (("L1", "glt"), "20220228", "183924", True, True),
        (("L1", "igm"), "20220228", "183924", False, False),
        (("L1", "igm"), "20220228", "183924", False, True),
        (("L1", "igm"), "20220228", "183924", True, True),
        (("L1", "obs"), "20220228", "183924", False, False),
        (("L1", "obs"), "20220228", "183924", False, True),
        (("L1", "obs"), "20220228", "183924", True, True),
        (("L1", "rdn"), "20220228", "183924", False, False),
        (("L1", "rdn"), "20220228", "183924", False, True),
        # (("L1", "rdn"), "20220228", "183924", True, True),
        # ("L2a", "20220228", "183924", True, False),
        ("L2a", "20220228", "183924", False, False),
        ("L2a", "20220228", "183924", False, True)
    ],
)

def test_intake_shift_driver(cat, band_dict, band_dict_filtered, dataset, date, time, ortho, filter_bad_bands):
    # Verify each type of dataset can be read in
    if isinstance(dataset, tuple):
        parent, child = dataset
        key = child
        dataset = getattr(getattr(cat, parent), child)()
    else:
        key = dataset
        dataset = getattr(cat, dataset)()
    
    ds = dataset(date=date, time=time, ortho=ortho, filter_bad_bands=filter_bad_bands).read_chunked()
    
    if filter_bad_bands:
        bands = band_dict_filtered 
        assert bands[key] == len(ds.wavelength)
    else:
        bands = band_dict 
        assert bands[key] == len(ds.wavelength)
        
    if ortho:
        glt = cat.L1.glt()(date=date, time=time, ortho=ortho).read_chunked()
        if key == 'glt':
            assert glt.values.shape == ds.values.shape
        else:
            y, x, z = glt.shape
            glt_array = glt.values.astype(int)
            valid_glt = np.all(glt_array != -9999, axis=-1)
            glt_array[valid_glt] -= 1 


            for x in range(valid_glt.shape[0]):
                if valid_glt[x,:].sum() == 0:
                    continue
                else:
                    break
            y = valid_glt[x, :]
            test_cord = glt_array[x, y, :][0]

            original_data = dataset(date=date, time=time, filter_bad_bands=filter_bad_bands).read_chunked()

            var = {'L2a': 'reflectance', 'rdn': 'radiance', 'obs': 'obs', 'igm': 'igm'}
            assert np.count_nonzero(np.nan_to_num(getattr(ds, var[key]).values,nan=-9999) != -9999.) ==  np.count_nonzero(valid_glt) * bands[key]
            assert (np.round(original_data.values[test_cord[1], test_cord[0]]) == np.round(getattr(ds, var[key]).values[x, y, :][0])).all()
