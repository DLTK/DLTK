from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import re
import os


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


class SaveableModule(AbstractModule):
    output_keys = []

    def __init__(self, name=None):
        self.input_placeholders = None
        self.saver = None
        super(SaveableModule, self).__init__(name)

    def _build_input_placeholder(self):
        raise NotImplementedError('Not implemented in abstract class')

    def save_metagraph(self, path, clear_devices=False, **kwargs):
        """

        Parameters
        ----------
        path : string
            path to save the metagraph to
        clear_devices : bool
            flag to toggle whether meta graph saves device placement of tensors
        kwargs
            additional arguments to the module build function

        """
        g = tf.get_default_graph()

        assert not g.finalized, 'Graph cannot be finalized'
        assert self.input_placeholders is not None, 'Input placeholders need to be built'

        self.saved_inputs = self.input_placeholders

        out = self._template(*self.saved_inputs, **kwargs)

        self.saved_outputs = out.values() if isinstance(out, dict) else [out]

        self.saved_var_list = list(self.get_variables(tf.GraphKeys.GLOBAL_VARIABLES))

        self.saver = tf.train.Saver(var_list=self.saved_var_list)

        g.clear_collection('saved_network')
        g.clear_collection('saved_inputs')
        g.clear_collection('saved_outputs')

        for i in self.saved_inputs:
            g.add_to_collections(['saved_inputs', 'saved_network'], i)

        for o in self.saved_outputs:
            g.add_to_collections(['saved_outputs', 'saved_network'], o)

        for tensor in self.saved_var_list:
            g.add_to_collection('saved_network', tensor)

        self.saver.export_meta_graph('{}.meta'.format(path), clear_devices=clear_devices)

        g.clear_collection('saved_network')
        g.clear_collection('saved_inputs')
        g.clear_collection('saved_outputs')

    def save_model(self, path, session):
        """Saves the network to a given path

        Parameters
        ----------
        path : string
            Path to the file to save the network in
        session : tf.Session
            Tensorflow Sessions holding the current variable states
        """

        assert self.saver is not None, 'Meta graph must be saved first'

        self.saver.save(session, path, write_meta_graph=False)

    @classmethod
    def load(cls, path, session):
        """

        Parameters
        ----------
        path : string
            Path to load the network from
        session : tf.Session
            Tensorflow Sessions to load the variables into

        Returns
        -------
        list : list of input placeholders saved
        list : list of outputs produced by the network

        """

        saver = tf.train.import_meta_graph('{}.meta'.format(path))

        saver.restore(session, path)

        inputs = tf.get_collection('saved_inputs')
        loaded_outputs = tf.get_collection('saved_outputs')

        outputs = {key: output for key, output in zip(cls.output_keys, loaded_outputs)}

        return inputs, outputs
