"""
Utility functions
"""
import numpy as np
from astropy.wcs import WCS
import ndcube
import astropy.units as u
from astropy.nddata import StdDevUncertainty

__all__ = ["quantity_1d_to_sequence"]


@u.quantity_input
def quantity_1d_to_sequence(intensity, wavelength: u.angstrom, uncertainty=None, meta=None):
    """
    Transform 1D `~astropy.units.Quantity` of intensities to a single-axis
    `~ndcube.NDCubeSequence`.

    This is a function for easily converting a 1D array of intensity values into a
    1D `~ndcube.NDCubeSequence` that can be passed to `sunkit_dem.Model`

    Parameters
    ----------
    intensities : `~astropy.units.Quantity`
    wavelengths : `~astropy.units.Quantity`
    uncertainty : `~astrpoy.units.Quantity`, optional
        Uncertainties on intensities
    meta : `dict` or `dict`-like, optional
    """
    cubes = []
    for j, (i, w) in enumerate(zip(intensity, wavelength)):
        if uncertainty is not None:
            _uncertainty = StdDevUncertainty(uncertainty.value[j, np.newaxis])
        else:
            _uncertainty = None
        wcs = {'CTYPE1': 'WAVE',
               'CUNIT1': w.unit.to_string(),
               'CDELT1': 1,
               'CRPIX1': 1,
               'CRVAL1': w.value,
               'NAXIS1': 1}
        cubes.append(ndcube.NDCube(
            i[np.newaxis],
            WCS(wcs),
            meta=meta,
            uncertainty=_uncertainty,
        ))
    return ndcube.NDCubeSequence(cubes, common_axis=0)
