"""
Base model classes for DEM models
"""
from abc import ABC, abstractmethod, abstractclassmethod
import copy

import numpy as np
import astropy.units as u
import astropy.wcs
import ndcube


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
    kernel : `dict`
        Each entry should be an `~astropy.units.Quantity` with a key that denotes
        the wavelength
    temperature : `~astropy.units.Quantity`
        Temperature bins over which to compute the DEM
    """

    _registry = dict()

    def __init_subclass__(cls, **kwargs):
        """
        An __init_subclass__ hook initializes all of the subclasses of a given class.
        So for each subclass, it will call this block of code on import.
        This replicates some metaclass magic without the need to be aware of metaclasses.
        Here we use this to register each subclass in a dict that has the
        `is_datasource_for` attribute.
        This is then passed into the Map Factory so we can register them.
        """
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'defines_model_for'):
            cls._registry[cls] = cls.defines_model_for

    @u.quantity_input
    def __init__(self, data, kernel, temperature: u.K, **kwargs):
        self._temperature = self._validate_temperature(temperature)
        self.data = self._validate_input_data(data)
        self.kernel = self._validate_kernel(kernel)

    def _validate_temperature(self, temperature):
        delta_log_temperature = np.diff(np.log10(temperature.to(u.K).value))
        if not all([d == delta_log_temperature[0] for d in delta_log_temperature]):
            raise ValueError('Temperature must be evenly spaced in log10')
        return temperature

    def _validate_input_data(self, data):
        """
        Check that input data is correctly formatted as an `ndcube.NDCubeSequence`
        """
        if not isinstance(data, ndcube.NDCubeSequence):
            raise ValueError('Input data must be an NDCubeSequence')
        if not all([hasattr(c, 'unit') for c in data]):
            raise u.UnitsError('All NDCube objects in NDCubeSequence must have units')
        # TODO: check that all WCS are the same, within some tolerance
        # NOTE: not always wavelength? Could be energy, filter wheel, etc.
        if data.cube_like_world_axis_physical_types[0] != 'em.wl':
            raise ValueError('The first axis of the sequence must be wavelength.')
        if data._common_axis != 0 or 'wavelength' not in data.common_axis_extra_coords:
            raise ValueError('Zeroth common axis must be wavelength.')
        return data

    def _validate_kernel(self, kernel):
        if len(kernel) != self.data.cube_like_dimensions[0].value:
            raise ValueError('Number of kernels must be equal to length of wavelength dimension.')
        if not all([k.shape == self.temperature.shape for k in kernel]):
            raise ValueError('Temperature and kernels must have the same shape.')
        return kernel

    @property
    @u.quantity_input
    def temperature(self) -> u.K:
        return self._temperature

    @property
    def wavelength(self):
        unit = self.data.wcs.to_header()[f'CUNIT{self.data.cube_like_dimensions.shape[0]}']
        return u.Quantity(self.data.common_axis_extra_coords['wavelength'], unit)

    @property
    def data_matrix(self):
        return u.Quantity(np.stack([n.data for n in self.data]), self.data[0].unit)

    @property
    def kernel_matrix(self):
        return u.Quantity(np.stack([self.kernel[w].value for w in self.wavelength.value]),
                          self.kernel[self.wavelength.value[0]].unit)

    def fit(self, **kwargs):
        result = self._model(**kwargs)
        wcs = self._make_dem_wcs()
        meta = self._make_dem_meta()
        return ndcube.NDCube(result, wcs, meta=meta)

    def _make_dem_wcs(self):
        wcs = self.data[0].wcs.to_header()  # This assumes that the WCS for all cubes is the same!
        n_axes = self.data.cube_like_dimensions.shape[0]
        wcs[f'CTYPE{n_axes}'] = 'LOG_TEMPERATURE'
        wcs[f'CUNIT{n_axes}'] = 'K'
        wcs[f'CDELT{n_axes}'] = np.diff(np.log10(self.temperature.to(u.K).value))[0]
        wcs[f'CRPIX{n_axes}'] = 1
        wcs[f'CRVAL{n_axes}'] = np.log10(self.temperature.to(u.K).value)[0]
        wcs[f'NAXIS{n_axes}'] = self.temperature.shape[0]
        for i in range(n_axes):
            wcs[f'NAXIS{i+1}'] = int(self.data.cube_like_dimensions[n_axes-1-i].value)

        return astropy.wcs.WCS(wcs)

    def _make_dem_meta(self):
        meta = copy.deepcopy(self.data[0].meta)
        # TODO: Remove/add keys?
        return meta
