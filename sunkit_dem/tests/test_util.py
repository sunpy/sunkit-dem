"""
Tests for utilities
"""
import ndcube
import numpy as np
import pytest

import astropy.units as u

from sunkit_dem.util import quantity_1d_to_collection

wavelengths = np.linspace(0, 1, 10) * u.angstrom
intensities = np.random.rand(wavelengths.shape[0]) * u.photon


@pytest.fixture
def collection1d():
    return quantity_1d_to_collection(intensities, wavelengths)


def test_is_collection(collection1d):
    assert isinstance(collection1d, ndcube.NDCollection)


def test_wavelengths(collection1d):
    wavelength_axis = u.Quantity([collection1d[k].axis_world_coords()[0] for k in collection1d.keys()]).squeeze()
    assert u.allclose(wavelength_axis, wavelengths)


def test_axis_type(collection1d):
    assert all([collection1d[k].array_axis_physical_types == [('em.wl',)] for k in collection1d.keys()])

def test_data(collection1d):
    collection_data = u.Quantity([collection1d[k].quantity for k in collection1d.keys()]).squeeze()
    assert u.allclose(intensities, collection_data)
