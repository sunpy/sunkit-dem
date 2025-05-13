"""
Factory for creating model classes
"""
from sunpy.util.datatype_factory_base import BasicRegistrationFactory

from .base_model import GenericModel

__all__ = ["Model"]


class ModelFactory(BasicRegistrationFactory):

    def __call__(self, *args, **kwargs):
        # TODO: account for duplicates?
        WidgetType = None
        for key in self.registry:
            if self.registry[key](*args, **kwargs):
                WidgetType = key
        # If no matches, return the default model
        WidgetType = self.default_widget_type if WidgetType is None else WidgetType
        return WidgetType(*args, **kwargs)


Model = ModelFactory(registry=GenericModel._registry,
                     default_widget_type=GenericModel,
                     additional_validation_functions=['defines_model_for'])
