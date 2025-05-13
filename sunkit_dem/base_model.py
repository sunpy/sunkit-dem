"""
Base model class for DEM models
"""
from abc import ABC, abstractmethod

import ndcube
import numpy as np
from ndcube.extra_coords.table_coord import MultipleTableCoordinate, QuantityTableCoordinate
from ndcube.wcs.wrappers import CompoundLowLevelWCS

import astropy.units as u
from astropy.nddata import StdDevUncertainty

__all__ = ["GenericModel"]


class BaseModel(ABC):

    @abstractmethod
    def _model(self):
        raise NotImplementedError

    def defines_model_for(self):
        raise NotImplementedError


class GenericModel(BaseModel):
    """
    Base class for implementing a differential emission measure model

    Parameters
    ----------
    data : `~ndcube.NDCubeSequence`
    kernel : `dict`
        `~astropy.units.Quantity` objects containing the kernels
        of each response. The keys should correspond to those in ``data``.
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
    def __init__(self, data, kernel, temperature_bin_edges: u.K, kernel_temperatures=None, **kwargs):
        self.temperature_bin_edges = temperature_bin_edges
        self.data = data
        self.kernel_temperatures = kernel_temperatures
        if self.kernel_temperatures is None:
            self.kernel_temperatures = self.temperature_bin_centers
        self.kernel = kernel

    @property
    def _keys(self):
        # Internal reference for entries in kernel and data
        # This ensures consistent ordering in kernel and data matrices
        return sorted(list(self.kernel.keys()))

    @property
    @u.quantity_input
    def temperature_bin_centers(self) -> u.K:
        return (self.temperature_bin_edges[1:] + self.temperature_bin_edges[:-1])/2.

    @property
    def data(self) -> ndcube.NDCollection:
        return self._data

    @data.setter
    def data(self, data):
        """
        Check that input data is correctly formatted as an
        `ndcube.NDCubeSequence`
        """
        if not isinstance(data, ndcube.NDCollection):
            raise ValueError('Input data must be an NDCollection')
        if not all([hasattr(data[k], 'unit') for k in data]):
            raise u.UnitsError('Each NDCube in NDCollection must have units')
        self._data = data

    @property
    def combined_mask(self):
        """
        Combined mask of all members of ``data``. Will be True if any member is masked.
        This is propagated to the final DEM result
        """
        combined_mask = []
        for k in self._keys:
            if self.data[k].mask is not None:
                combined_mask.append(self.data[k].mask)
            else:
                combined_mask.append(np.full(self.data[k].data.shape, False))
        return np.any(combined_mask, axis=0)

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, kernel):
        if len(kernel) != len(self.data):
            raise ValueError('Number of kernels must be equal to length of wavelength dimension.')
        if not all([v.shape == self.kernel_temperatures.shape for _, v in kernel.items()]):
            raise ValueError('Temperature bin centers and kernels must have the same shape.')
        self._kernel = kernel

    @property
    def data_matrix(self):
        return np.stack([self.data[k].data for k in self._keys])

    @property
    def kernel_matrix(self):
        return np.stack([self.kernel[k].value for k in self._keys])

    def fit(self, *args, **kwargs):
        r"""
        Apply inversion procedure to data.

        Returns
        -------
        dem : `~ndcube.NDCube`
            Differential emission measure as a function of temperature. The
            temperature axis is evenly spaced in :math:`\log{T}`. The number
            of dimensions depend on the input data.
        """
        dem_dict = self._model(*args, **kwargs)
        wcs = self._make_dem_wcs()
        meta = self._make_dem_meta()
        dem_data = dem_dict.pop('dem')
        mask = np.full(dem_data.shape, False)
        mask[:,...] = self.combined_mask
        dem = ndcube.NDCube(dem_data,
                            wcs,
                            meta=meta,
                            mask=mask,
                            uncertainty=StdDevUncertainty(dem_dict.pop('uncertainty')))
        cubes = [('dem', dem),]
        for k in dem_dict:
            cubes += [(k, ndcube.NDCube(dem_dict[k], wcs, meta=meta))]
        return ndcube.NDCollection(cubes, )

    def _make_dem_wcs(self):
        data_wcs = self.data[self._keys[0]].wcs
        temp_table = QuantityTableCoordinate(self.temperature_bin_centers,
                                             names='temperature',
                                             physical_types='phys.temperature')
        temp_table_coord = MultipleTableCoordinate(temp_table)
        mapping = list(range(data_wcs.pixel_n_dim))
        mapping.extend([data_wcs.pixel_n_dim] * temp_table_coord.wcs.pixel_n_dim)
        compound_wcs = CompoundLowLevelWCS(data_wcs, temp_table_coord.wcs, mapping=mapping)
        return compound_wcs

    def _make_dem_meta(self):
        # Individual classes should override this if they want specific metadata
        return {}
