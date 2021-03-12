"""
Base model class for DEM models
"""
from abc import ABC, abstractmethod, abstractclassmethod
import copy

import numpy as np
import astropy.units as u
import astropy.wcs
import ndcube

__all__ = ["GenericModel"]


class BaseModel(ABC):

    @abstractmethod
    def _model(self):
        raise NotImplementedError

    @abstractclassmethod
    def defines_model_for(self):
        raise NotImplementedError


class GenericModel(BaseModel):
    """
    Base class for implementing a differential emission measure model

    Parameters
    ----------
    data : `~ndcube.NDCubeSequence`
    kernel : `list`
        List of `~astropy.units.Quantity` objects containing the kernels
        of each response. The order must be the same as the wavelength
        sequence ordering of `data`.
    temperature_bin_edges : `~astropy.units.Quantity`
        Edges of the temperature bins in which the DEM is computed. The
        rightmost edge is included. The kernel is evaluated at the bin centers.
        The bin widths must be equal in log10.
    """

    _registry = dict()

    def __init_subclass__(cls, **kwargs):
        """
        An __init_subclass__ hook initializes all of the subclasses of a given
        class. So for each subclass, it will call this block of code on import.
        This replicates some metaclass magic without the need to be aware of
        metaclasses. Here we use this to register each subclass in a dict that
        has the `defines_model_for` attribute. This is then passed into the Map
        Factory so we can register them.
        """
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'defines_model_for'):
            cls._registry[cls] = cls.defines_model_for

    @u.quantity_input
    def __init__(self, data, kernel, temperature_bin_edges: u.K, **kwargs):
        self.temperature_bin_edges = temperature_bin_edges
        self.data = data
        self.kernel = kernel

    @property
    @u.quantity_input
    def temperature_bin_edges(self) -> u.K:
        return self._temperature_bin_edges

    @temperature_bin_edges.setter
    @u.quantity_input
    def temperature_bin_edges(self, temperature_bin_edges: u.K):
        delta_log_temperature = np.diff(np.log10(temperature_bin_edges.to(u.K).value))
        # NOTE: Very small numerical differences not important
        # NOTE: Can be removed if we use gWCS to create the resulting DEM cube
        if not np.allclose(delta_log_temperature, delta_log_temperature[0], atol=1e-10, rtol=0):
            raise ValueError('Temperature must be evenly spaced in log10')
        self._temperature_bin_edges = temperature_bin_edges

    @property
    @u.quantity_input
    def temperature_bin_centers(self) -> u.K:
        log_temperature = np.log10(self.temperature_bin_edges.value)
        log_temperature_centers = (log_temperature[1:] + log_temperature[:-1])/2.
        return u.Quantity(10.**log_temperature_centers, self.temperature_bin_edges.unit)

    @property
    def data(self) -> ndcube.NDCubeSequence:
        return self._data

    @data.setter
    def data(self, data):
        """
        Check that input data is correctly formatted as an
        `ndcube.NDCubeSequence`
        """
        if not isinstance(data, ndcube.NDCubeSequence):
            raise ValueError('Input data must be an NDCubeSequence')
        if not all([hasattr(c, 'unit') for c in data]):
            raise u.UnitsError('Each NDCube in NDCubeSequence must have units')
        # TODO: check that all WCS are the same, within some tolerance
        # NOTE: not always wavelength? Could be energy, filter wheel, etc.
        if data.cube_like_array_axis_physical_types[0] != ('em.wl',):
            raise ValueError('First axis of sequence must be wavelength.')
        self._data = data

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, kernel):
        if len(kernel) != self.data.cube_like_dimensions[0].value:
            raise ValueError('Number of kernels must be equal to length of wavelength dimension.')
        if not all([k.shape == self.temperature_bin_centers.shape for k in kernel]):
            raise ValueError('Temperature and kernels must have the same shape.')
        self._kernel = kernel

    @property
    def wavelength(self):
        # TODO: handle non-numeric "wavelength" designations
        return u.Quantity(self.data.common_axis_coords[0], self.data[0].wcs.wcs.cunit[-1])

    @property
    def data_matrix(self):
        return u.Quantity(np.stack([n.data.squeeze() for n in self.data]), self.data[0].unit)

    @property
    def kernel_matrix(self):
        return u.Quantity(np.stack([k.value for k in self.kernel]), self.kernel[0].unit)

    def fit(self, *args, **kwargs):
        """
        Apply inversion procedure to data.

        Returns
        -------
        dem : `~ndcube.NDCube`
            Differential emission measure as a function of temperature. The
            temperature axis is evenly spaced in :math:`\log{T}`. The number
            of dimensions depend on the input data.
        """
        dem, uncertainty = self._model(*args, **kwargs)
        wcs = self._make_dem_wcs()
        meta = self._make_dem_meta()
        # NOTE: Bug in NDData that does not allow passing quantity as uncertainty
        uncertainty = uncertainty.value if isinstance(uncertainty, u.Quantity) else uncertainty
        return ndcube.NDCube(dem, wcs, meta=meta, uncertainty=uncertainty,)

    def _make_dem_wcs(self):
        # NOTE: Assumes that WCS for all cubes is the same
        wcs = self.data[0].wcs.to_header()
        n_axes = self.data[0].wcs.naxis
        wcs[f'CTYPE{n_axes}'] = 'LOG_TEMPERATURE'
        wcs[f'CUNIT{n_axes}'] = 'K'
        wcs[f'CDELT{n_axes}'] = np.diff(np.log10(self.temperature_bin_centers.to(u.K).value))[0]
        wcs[f'CRPIX{n_axes}'] = 1
        wcs[f'CRVAL{n_axes}'] = np.log10(self.temperature_bin_centers.to(u.K).value)[0]
        wcs[f'NAXIS{n_axes}'] = self.temperature_bin_centers.shape[0]
        for i in range(n_axes-1):
            # FIXME: better way to get this info than internal var?
            wcs[f'NAXIS{i+1}'] = self.data[0].wcs._naxis[i]

        return astropy.wcs.WCS(wcs)

    def _make_dem_meta(self):
        meta = copy.deepcopy(self.data[0].meta)
        # TODO: Remove/add keys?
        return meta
