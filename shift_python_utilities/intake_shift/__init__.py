# from intake_xarray._version import get_versions
# __version__ = get_versions()['version']
# del get_versions
import intake  # Import this first to avoid circular imports during discovery.
from intake.container import register_container
from .shift_xarray import ShiftXarray
from .shift_catalog import SHIFTCatalog as shift_catalog
from .bad_bands import bad_bands

try:
    intake.register_driver('SHIFT-xarray', ShiftXarray)
except ValueError:
    pass

register_container('SHIFT-xarray', ShiftXarray)
