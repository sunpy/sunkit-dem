"""
Tests for utilities
"""
import numpy as np
import pytest

import astropy.units as u

from sunkit_dem.util import quantity_1d_to_sequence

wavelengths = np.linspace(0, 1, 10) * u.angstrom
intensities = np.random.rand(wavelengths.shape[0]) * u.ct


@pytest.fixture
def sequence1d():
    return quantity_1d_to_sequence(intensities, wavelengths)


def test_dimensions(sequence1d):
    assert all(sequence1d.cube_like_shape == [10]*u.pix)


def test_common_axis(sequence1d):
    common_axis = u.Quantity(sequence1d.common_axis_coords[0])
    assert common_axis.shape == wavelengths.shape
    assert u.allclose(common_axis, wavelengths, rtol=1e-10)


def test_axis_type(sequence1d):
    assert sequence1d.cube_like_array_axis_physical_types == [('em.wl',)]
