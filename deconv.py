import functools
import logging
import numpy as np

from theano.compat import OrderedDict
from theano import tensor as T

from pylearn2.linear.conv2d_c01b import make_random_conv2D
from pylearn2.models import Model
from pylearn2.models.maxout import check_cuda # TODO: import from original path
from pylearn2.models.mlp import Layer
#from pylearn2.models.maxout import py_integer_types # TODO: import from orig path
from pylearn2.space import Conv2DSpace
from pylearn2.utils import sharedX

logger = logging.getLogger(__name__)

class Deconv(Layer):
    def __init__(self,
                 num_channels,
                 kernel_shape,
                 layer_name,
                 irange=None,
                 init_bias=0.,
                 W_lr_scale=None,
                 b_lr_scale=None,
                 pad_out=0,
                 fix_kernel_shape=False,
                 partial_sum=1,
                 tied_b=False,
                 max_kernel_norm=None,
                 output_stride=(1, 1)):
        check_cuda(str(type(self)))
        super(Deconv, self).__init__()

        detector_channels = num_channels

        self.__dict__.update(locals())
        del self.self

    @functools.wraps(Model.get_lr_scalers)
    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    def set_input_space(self, space):
        """
        Tells the layer to use the specified input space.

        This resets parameters! The kernel tensor is initialized with the
        size needed to receive input from this space.

        Parameters
        ----------
        space : Space
            The Space that the input will lie in.
        """

        setup_deconv_detector_layer_c01b(layer=self,
                                  input_space=space,
                                  rng=self.mlp.rng)

        rng = self.mlp.rng

        detector_shape = self.detector_space.shape


        self.output_space = self.detector_space

        logger.info('Output space: {0}'.format(self.output_space.shape))

    def _modify_updates(self, updates):
        """
        Replaces the values in `updates` if needed to enforce the options set
        in the __init__ method, including `max_kernel_norm`.

        Parameters
        ----------
        updates : OrderedDict
            A dictionary mapping parameters (including parameters not
            belonging to this model) to updated values of those parameters.
            The dictionary passed in contains the updates proposed by the
            learning algorithm. This function modifies the dictionary
            directly. The modified version will be compiled and executed
            by the learning algorithm.
        """

        if self.max_kernel_norm is not None:
            W, = self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=(0, 1, 2)))
                desired_norms = T.clip(row_norms, 0, self.max_kernel_norm)
                scales = desired_norms / (1e-7 + row_norms)
                updates[W] = (updated_W * scales.dimshuffle('x', 'x', 'x', 0))

    @functools.wraps(Model.get_params)
    def get_params(self):
        assert self.b.name is not None
        W, = self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        return rval

    @functools.wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    @functools.wraps(Layer.set_weights)
    def set_weights(self, weights):
        W, = self.transformer.get_params()
        W.set_value(weights)

    @functools.wraps(Layer.set_biases)
    def set_biases(self, biases):
        self.b.set_value(biases)

    @functools.wraps(Layer.get_biases)
    def get_biases(self):
        return self.b.get_value()

    @functools.wraps(Model.get_weights_topo)
    def get_weights_topo(self):
        return self.transformer.get_weights_topo()

    @functools.wraps(Layer.get_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None, state=None, targets=None):

        W, = self.transformer.get_params()

        assert W.ndim == 4

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=(0, 1, 2)))

        P = state

        rval = OrderedDict()

        vars_and_prefixes = [(P, '')]

        for var, prefix in vars_and_prefixes:
            if not hasattr(var, 'ndim') or var.ndim != 4:
                print "expected 4D tensor, got "
                print var
                print type(var)
                if isinstance(var, tuple):
                    print "tuple length: ", len(var)
                assert False
            v_max = var.max(axis=(1, 2, 3))
            v_min = var.min(axis=(1, 2, 3))
            v_mean = var.mean(axis=(1, 2, 3))
            v_range = v_max - v_min

            # max_x.mean_u is "the mean over *u*nits of the max over
            # e*x*amples" The x and u are included in the name because
            # otherwise its hard to remember which axis is which when reading
            # the monitor I use inner.outer rather than outer_of_inner or
            # something like that because I want mean_x.* to appear next to
            # each other in the alphabetical list, as these are commonly
            # plotted together
            for key, val in [('max_x.max_u',    v_max.max()),
                             ('max_x.mean_u',   v_max.mean()),
                             ('max_x.min_u',    v_max.min()),
                             ('min_x.max_u',    v_min.max()),
                             ('min_x.mean_u',   v_min.mean()),
                             ('min_x.min_u',    v_min.min()),
                             ('range_x.max_u',  v_range.max()),
                             ('range_x.mean_u', v_range.mean()),
                             ('range_x.min_u',  v_range.min()),
                             ('mean_x.max_u',   v_mean.max()),
                             ('mean_x.mean_u',  v_mean.mean()),
                             ('mean_x.min_u',   v_mean.min())]:
                rval[prefix+key] = val

        rval.update(OrderedDict([('kernel_norms_min',  row_norms.min()),
                            ('kernel_norms_mean', row_norms.mean()),
                            ('kernel_norms_max',  row_norms.max()), ]))

        return rval

    @functools.wraps(Layer.fprop)
    def fprop(self, state_below):
        check_cuda(str(type(self)))

        self.input_space.validate(state_below)

        z = self.transformer.lmul_T(state_below)

        self.output_space.validate(z)

        if not hasattr(self, 'tied_b'):
            self.tied_b = False
        if self.tied_b:
            b = self.b.dimshuffle(0, 'x', 'x', 'x')
        else:
            b = self.b.dimshuffle(0, 1, 2, 'x')

        return z + b



def setup_deconv_detector_layer_c01b(layer, input_space, rng, irange="not specified"):
    """
    layer. This function sets up only the detector layer.

    Does the following:

    * raises a RuntimeError if cuda is not available
    * sets layer.input_space to input_space
    * sets up addition of dummy channels for compatibility with cuda-convnet:

      - layer.dummy_channels: # of dummy channels that need to be added
        (You might want to check this and raise an Exception if it's not 0)
      - layer.dummy_space: The Conv2DSpace representing the input with dummy
        channels added

    * sets layer.detector_space to the space for the detector layer
    * sets layer.transformer to be a Conv2D instance
    * sets layer.b to the right value

    Parameters
    ----------
    layer : object
        Any python object that allows the modifications described below and
        has the following attributes:

          * pad : int describing amount of zero padding to add
          * kernel_shape : 2-element tuple or list describing spatial shape of
            kernel
          * fix_kernel_shape : bool, if true, will shrink the kernel shape to
            make it feasible, as needed (useful for hyperparameter searchers)
          * detector_channels : The number of channels in the detector layer
          * init_bias : numeric constant added to a tensor of zeros to
            initialize the bias
          * tied_b : If true, biases are shared across all spatial locations
    input_space : WRITEME
        A Conv2DSpace to be used as input to the layer
    rng : WRITEME
        A numpy RandomState or equivalent
    """

    if irange != "not specified":
        raise AssertionError(
            "There was a bug in setup_detector_layer_c01b."
            "It uses layer.irange instead of the irange parameter to the "
            "function. The irange parameter is now disabled by this "
            "AssertionError, so that this error message can alert you that "
            "the bug affected your code and explain why the interface is "
            "changing. The irange parameter to the function and this "
            "error message may be removed after April 21, 2014."
        )

    # Use "self" to refer to layer from now on, so we can pretend we're
    # just running in the set_input_space method of the layer
    self = layer

    # Make sure cuda is available
    check_cuda(str(type(self)))

    # Validate input
    if not isinstance(input_space, Conv2DSpace):
        raise TypeError("The input to a convolutional layer should be a "
                        "Conv2DSpace, but layer " + self.layer_name + " got " +
                        str(type(self.input_space)))

    if not hasattr(self, 'detector_channels'):
        raise ValueError("layer argument must have a 'detector_channels' "
                         "attribute specifying how many channels to put in "
                         "the convolution kernel stack.")

    # Store the input space
    self.input_space = input_space

    # Make sure number of channels is supported by cuda-convnet
    # (multiple of 4 or <= 3)
    # If not supported, pad the input with dummy channels
    ch = self.detector_channels
    rem = ch % 4
    if ch > 3 and rem != 0:
        raise NotImplementedError("Need to do dummy channels on the output")
    #    self.dummy_channels = 4 - rem
    #else:
    #    self.dummy_channels = 0
    #self.dummy_space = Conv2DSpace(
    #    shape=input_space.shape,
    #    channels=input_space.num_channels + self.dummy_channels,
    #    axes=('c', 0, 1, 'b')
    #)

    if hasattr(self, 'output_stride'):
        kernel_stride = self.output_stride
    else:
        assert False # not sure if I got the name right, remove this assert if I did
        kernel_stride = [1, 1]


    #o_sh = int(np.ceil((i_sh + 2. * self.pad - k_sh) / float(k_st))) + 1
    #o_sh -1 = np.ceil((i_sh + 2. * self.pad - k_sh) / float(k_st))
    #inv_ceil(o_sh -1) = (i_sh + 2. * self.pad - k_sh) / float(k_st)
    #float(k_st) inv_cel(o_sh -1) = (i_sh + 2 * self.pad -k_sh)
    # i_sh = k_st inv_ceil(o_sh-1) - 2 * self.pad + k_sh

    output_shape = \
        [k_st * (i_sh - 1) - 2 * self.pad_out + k_sh
         for i_sh, k_sh, k_st in zip(self.input_space.shape,
                                     self.kernel_shape, kernel_stride)]


    if self.input_space.num_channels < 16:
        raise ValueError("Cuda-convnet requires the input to lmul_T to have "
                         "at least 16 channels.")

    self.detector_space = Conv2DSpace(shape=output_shape,
                                      num_channels=self.detector_channels,
                                      axes=('c', 0, 1, 'b'))

    if hasattr(self, 'partial_sum'):
        partial_sum = self.partial_sum
    else:
        partial_sum = 1

    if hasattr(self, 'sparse_init') and self.sparse_init is not None:
        self.transformer = \
            checked_call(make_sparse_random_conv2D,
                         OrderedDict([('num_nonzero', self.sparse_init),
                                      ('input_space', self.detector_space),
                                      ('output_space', self.input_space),
                                      ('kernel_shape', self.kernel_shape),
                                      ('pad', self.pad),
                                      ('partial_sum', partial_sum),
                                      ('kernel_stride', kernel_stride),
                                      ('rng', rng)]))
    else:
        self.transformer = make_random_conv2D(
            irange=self.irange,
            input_axes=self.detector_space.axes,
            output_axes=self.input_space.axes,
            input_channels=self.detector_space.num_channels,
            output_channels=self.input_space.num_channels,
            kernel_shape=self.kernel_shape,
            pad=self.pad_out,
            partial_sum=partial_sum,
            kernel_stride=kernel_stride,
            rng=rng,
            input_shape=self.detector_space.shape
        )

    W, = self.transformer.get_params()
    W.name = self.layer_name + '_W'

    if self.tied_b:
        self.b = sharedX(np.zeros(self.detector_space.num_channels) +
                         self.init_bias)
    else:
        self.b = sharedX(self.detector_space.get_origin() + self.init_bias)
    self.b.name = self.layer_name + '_b'

    logger.info('Input shape: {0}'.format(self.input_space.shape))
    print layer.layer_name + ' detector space: {0}'.format(self.detector_space.shape)
