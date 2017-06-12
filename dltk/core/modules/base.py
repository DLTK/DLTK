from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import re


class AbstractModule(object):
    """Superclass for DLTK core modules - strongly inspired by Sonnet: https://github.com/deepmind/sonnet

    This class wraps implements a wrapping of `tf.make_template` for automatic variable sharing. Each subclass needs to
    implement a `_build` function used for the template and call this superclass' `__init__` to create the template.
    For the variable sharing to work, variables inside `_build` have to be created via `tf.get_variable` instead of
    `tf.Variable`.

    The created template is automatically called using `__call__`.
    """
    MODEL_COLLECTIONS = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MODEL_VARIABLES]
    TRAINABLE_COLLECTIONS = MODEL_COLLECTIONS + [tf.GraphKeys.TRAINABLE_VARIABLES]
    WEIGHT_COLLECTIONS = TRAINABLE_COLLECTIONS + [tf.GraphKeys.WEIGHTS]
    BIAS_COLLECTIONS = TRAINABLE_COLLECTIONS + [tf.GraphKeys.BIASES]
    MOVING_COLLECTIONS = MODEL_COLLECTIONS + [tf.GraphKeys.MOVING_AVERAGE_VARIABLES]

    def __init__(self, name=None):
        """Initialisation of the template and naming of the module

        Parameters
        ----------
        name : string
            name of the module
        """
        self.name = name
        self.variables = []
        self._template = tf.make_template(name, self._build, create_scope_now_=True)

        # Update __call__ and the object docstrings to enable better introspection (from Sonnet)
        self.__doc__ = self._build.__doc__
        self.__call__.__func__.__doc__ = self._build.__doc__

    def _build(self, *args, **kwargs):
        """Abstract function that is use to make the template when building the module

        Raises
        -------
        NotImplementedError
            This is an abstract function

        """
        raise NotImplementedError('Not implemented in abstract class')

    def __call__(self, *args, **kwargs):
        """Wrapper to call template when module is called

        Returns
        -------
        object
            Returns output of _build function

        """
        out = self._template(*args, **kwargs)
        return out

    @property
    def variable_scope(self):
        """Getter to access variable scope of the built template"""
        return self._template.variable_scope

    def get_variables(self, collection=tf.GraphKeys.TRAINABLE_VARIABLES):
        """Helper to get all variables of a given collection created within this module

        Parameters
        ----------
        collection : string, optional
            Identifier of the collection to get variables from. Defaults to `tf.GraphKeys.TRAINABLE_VARIABLES`

        Returns
        -------
        list
            List of `tf.Variables` that are part of the collection and within the scope of this module
        """
        scope_name = re.escape(self.variable_scope.name) + "/"
        return tuple(tf.get_collection(collection, scope_name))