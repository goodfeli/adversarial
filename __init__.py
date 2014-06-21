"""
Code for "Generative Adversarial Networks". Please cite the ArXiv paper in
any published research work making use of this code.
"""
import functools
wraps = functools.wraps
import itertools
import numpy
np = numpy
import theano
import warnings

from theano.compat import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor as T

from pylearn2.space import VectorSpace
from pylearn2.costs.cost import Cost
from pylearn2.costs.cost import DefaultDataSpecsMixin
from pylearn2.models.mlp import Layer
from pylearn2.models.mlp import Linear
from pylearn2.models import Model
from pylearn2.space import CompositeSpace
from pylearn2.train_extensions import TrainExtension
from pylearn2.utils import block_gradient
from pylearn2.utils import safe_zip
from pylearn2.utils import serial
from pylearn2.utils import sharedX

class AdversaryPair(Model):

    def __init__(self, generator, discriminator, inferer=None,
                 inference_monitoring_batch_size=128,
                 monitor_generator=True,
                 monitor_discriminator=True,
                 monitor_inference=True,
                 shrink_d = 0.):
        Model.__init__(self)
        self.__dict__.update(locals())
        del self.self

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'inferer' not in state:
            self.inferer = None
        if 'inference_monitoring_batch_size' not in state:
            self.inference_monitoring_batch_size = 128  # TODO: HACK
        if 'monitor_generator' not in state:
            self.monitor_generator = True
        if 'monitor_discriminator' not in state:
            self.monitor_discriminator = True
        if 'monitor_inference' not in state:
            self.monitor_inference = True

    def get_params(self):
        p = self.generator.get_params() + self.discriminator.get_params()
        if hasattr(self, 'inferer') and self.inferer is not None:
            p += self.inferer.get_params()
        return p

    def get_input_space(self):
        return self.discriminator.get_input_space()

    def get_weights_topo(self):
        return self.discriminator.get_weights_topo()

    def get_weights(self):
        return self.discriminator.get_weights()

    def get_weights_format(self):
        return self.discriminator.get_weights_format()

    def get_weights_view_shape(self):
        return self.discriminator.get_weights_view_shape()

    def get_monitoring_channels(self, data):
        rval = OrderedDict()

        g_ch = self.generator.get_monitoring_channels(data)
        d_ch = self.discriminator.get_monitoring_channels((data, None))
        samples = self.generator.sample(100)
        d_samp_ch = self.discriminator.get_monitoring_channels((samples, None))

        i_ch = OrderedDict()
        if self.inferer is not None:
            batch_size = self.inference_monitoring_batch_size
            sample, noise, _ = self.generator.sample_and_noise(batch_size)
            i_ch.update(self.inferer.get_monitoring_channels((sample, noise)))

        if self.monitor_generator:
            for key in g_ch:
                rval['gen_' + key] = g_ch[key]
        if self.monitor_discriminator:
            for key in d_ch:
                rval['dis_on_data_' + key] = d_samp_ch[key]
            for key in d_ch:
                rval['dis_on_samp_' + key] = d_ch[key]
        if self.monitor_inference:
            for key in i_ch:
                rval['inf_' + key] = i_ch[key]
        return rval

    def get_monitoring_data_specs(self):

        space = self.discriminator.get_input_space()
        source = self.discriminator.get_input_source()
        return (space, source)

    def _modify_updates(self, updates):
        self.generator.modify_updates(updates)
        self.discriminator.modify_updates(updates)
        if self.shrink_d != 0.:
            for param in self.discriminator.get_params():
                if param in updates:
                    updates[param] = self.shrink_d * updates[param]
        if self.inferer is not None:
            self.inferer.modify_updates(updates)

    def get_lr_scalers(self):

        rval = self.generator.get_lr_scalers()
        rval.update(self.discriminator.get_lr_scalers())
        return rval

def add_layers(mlp, pretrained, start_layer=0):
    model = serial.load(pretrained)
    pretrained_layers = model.generator.mlp.layers
    assert pretrained_layers[start_layer].get_input_space() == mlp.layers[-1].get_output_space()
    mlp.layers.extend(pretrained_layers[start_layer:])
    return mlp



class Generator(Model):

    def __init__(self, mlp, noise = "gaussian", monitor_ll = False, ll_n_samples = 100, ll_sigma = 0.2):
        Model.__init__(self)
        self.__dict__.update(locals())
        del self.self
        self.theano_rng = MRG_RandomStreams(2014 * 5 + 27)

    def get_input_space(self):
        return self.mlp.get_input_space()

    def sample_and_noise(self, num_samples, default_input_include_prob=1., default_input_scale=1., all_g_layers=False):
        n = self.mlp.get_input_space().get_total_dimension()
        noise = self.get_noise((num_samples, n))
        formatted_noise = VectorSpace(n).format_as(noise, self.mlp.get_input_space())
        if all_g_layers:
            rval = self.mlp.dropout_fprop(formatted_noise, default_input_include_prob=default_input_include_prob, default_input_scale=default_input_scale, return_all=all_g_layers)
            other_layers, rval = rval[:-1], rval[-1]
        else:
            rval = self.mlp.dropout_fprop(formatted_noise, default_input_include_prob=default_input_include_prob, default_input_scale=default_input_scale)
            other_layers = None
        return rval, formatted_noise, other_layers

    def sample(self, num_samples, default_input_include_prob=1., default_input_scale=1.):
        sample, _, _ = self.sample_and_noise(num_samples, default_input_include_prob, default_input_scale)
        return sample

    def inpainting_sample_and_noise(self, X, default_input_include_prob=1., default_input_scale=1.):
        # Very hacky! Specifically for inpainting right half of CIFAR-10 given left half
        # assumes X is b01c
        assert X.ndim == 4
        input_space = self.mlp.get_input_space()
        n = input_space.get_total_dimension()
        image_size = input_space.shape[0]
        half_image = int(image_size / 2)
        data_shape = (X.shape[0], image_size, half_image, input_space.num_channels)

        noise = self.theano_rng.normal(size=data_shape, dtype='float32')
        Xg = T.set_subtensor(X[:,:,half_image:,:], noise)
        sampled_part, noise =  self.mlp.dropout_fprop(Xg, default_input_include_prob=default_input_include_prob, default_input_scale=default_input_scale), noise
        sampled_part = sampled_part.reshape(data_shape)
        rval = T.set_subtensor(X[:, :, half_image:, :], sampled_part)
        return rval, noise


    def get_monitoring_channels(self, data):
        if data is None:
            m = 100
        else:
            m = data.shape[0]
        n = self.mlp.get_input_space().get_total_dimension()
        noise = self.get_noise((m, n))
        rval = OrderedDict()

        try:
            rval.update(self.mlp.get_monitoring_channels((noise, None)))
        except Exception:
            warnings.warn("something went wrong with generator.mlp's monitoring channels")

        if  self.monitor_ll:
            rval['ll'] = T.cast(self.ll(data, self.ll_n_samples, self.ll_sigma),
                                        theano.config.floatX).mean()
            rval['nll'] = -rval['ll']
        return rval

    def get_noise(self, size):

        # Allow just requesting batch size
        if isinstance(size, int):
            size = (size, self.get_input_space().get_total_dimension())

        if not hasattr(self, 'noise'):
            self.noise = "gaussian"
        if self.noise == "uniform":
            return self.theano_rng.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=size, dtype='float32')
        elif self.noise == "gaussian":
            return self.theano_rng.normal(size=size, dtype='float32')
        elif self.noise == "spherical":
            noise = self.theano_rng.normal(size=size, dtype='float32')
            noise = noise / T.maximum(1e-7, T.sqrt(T.sqr(noise).sum(axis=1))).dimshuffle(0, 'x')
            return noise
        else:
            raise NotImplementedError(self.noise)

    def get_params(self):
        return self.mlp.get_params()

    def get_output_space(self):
        return self.mlp.get_output_space()

    def ll(self, data, n_samples, sigma):

        samples = self.sample(n_samples)
        output_space = self.mlp.get_output_space()
        if 'Conv2D' in str(output_space):
            samples = output_space.convert(samples, output_space.axes, ('b', 0, 1, 'c'))
            samples = samples.flatten(2)
            data = output_space.convert(data, output_space.axes, ('b', 0, 1, 'c'))
            data = data.flatten(2)
        parzen = theano_parzen(data, samples, sigma)
        return parzen

    def _modify_updates(self, updates):
        self.mlp.modify_updates(updates)

    def get_lr_scalers(self):
        return self.mlp.get_lr_scalers()

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'monitor_ll' not in state:
            self.monitor_ll = False


class IntrinsicDropoutGenerator(Generator):
    def __init__(self, default_input_include_prob, default_input_scale,
                        input_include_probs=None, input_scales=None, **kwargs):
        super(IntrinsicDropoutGenerator, self).__init__(**kwargs)
        self.__dict__.update(locals())
        del self.self

    def sample_and_noise(self, num_samples, default_input_include_prob=1., default_input_scale=1., all_g_layers=False):
        if all_g_layers:
            raise NotImplementedError()
        n = self.mlp.get_input_space().get_total_dimension()
        noise = self.theano_rng.normal(size=(num_samples, n), dtype='float32')
        formatted_noise = VectorSpace(n).format_as(noise, self.mlp.get_input_space())
        # ignores dropout args
        default_input_include_prob = self.default_input_include_prob
        default_input_scale = self.default_input_scale
        input_include_probs = self.input_include_probs
        input_scales = self.input_scales
        return self.mlp.dropout_fprop(formatted_noise,
                                      default_input_include_prob=default_input_include_prob,
                                      default_input_scale=default_input_scale,
                                      input_include_probs=input_include_probs,
                                      input_scales=input_scales), formatted_noise, None

class AdversaryCost2(DefaultDataSpecsMixin, Cost):
    """
    """

    # Supplies own labels, don't get them from the dataset
    supervised = False

    def __init__(self, scale_grads=1, target_scale=.1,
            discriminator_default_input_include_prob = 1.,
            discriminator_input_include_probs=None,
            discriminator_default_input_scale=1.,
            discriminator_input_scales=None,
            generator_default_input_include_prob = 1.,
            generator_default_input_scale=1.,
            inference_default_input_include_prob=None,
            inference_input_include_probs=None,
            inference_default_input_scale=1.,
            inference_input_scales=None,
            init_now_train_generator=True,
            ever_train_discriminator=True,
            ever_train_generator=True,
            ever_train_inference=True,
            no_drop_in_d_for_g=False,
            alternate_g = False,
            infer_layer=None,
            noise_both = 0.,
            blend_obj = False,
            minimax_coeff = 1.,
            zurich_coeff = 1.):
        self.__dict__.update(locals())
        del self.self
        # These allow you to dynamically switch off training parts.
        # If the corresponding ever_train_* is False, these have
        # no effect.
        self.now_train_generator = sharedX(init_now_train_generator)
        self.now_train_discriminator = sharedX(numpy.array(1., dtype='float32'))
        self.now_train_inference = sharedX(numpy.array(1., dtype='float32'))

    def expr(self, model, data, **kwargs):
        S, d_obj, g_obj, i_obj = self.get_samples_and_objectives(model, data)
        l = []
        # This stops stuff from ever getting computed if we're not training
        # it.
        if self.ever_train_discriminator:
            l.append(d_obj)
        if self.ever_train_generator:
            l.append(g_obj)
        if self.ever_train_inference:
            l.append(i_obj)
        return sum(l)

    def get_samples_and_objectives(self, model, data):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        assert isinstance(model, AdversaryPair)
        g = model.generator
        d = model.discriminator

        # Note: this assumes data is design matrix
        X = data
        m = data.shape[space.get_batch_axis()]
        y1 = T.alloc(1, m, 1)
        y0 = T.alloc(0, m, 1)
        # NOTE: if this changes to optionally use dropout, change the inference
        # code below to use a non-dropped-out version.
        S, z, other_layers = g.sample_and_noise(m, default_input_include_prob=self.generator_default_input_include_prob, default_input_scale=self.generator_default_input_scale, all_g_layers=(self.infer_layer is not None))

        if self.noise_both != 0.:
            rng = MRG_RandomStreams(2014 / 6 + 2)
            S = S + rng.normal(size=S.shape, dtype=S.dtype) * self.noise_both
            X = X + rng.normal(size=X.shape, dtype=S.dtype) * self.noise_both

        y_hat1 = d.dropout_fprop(X, self.discriminator_default_input_include_prob,
                                     self.discriminator_input_include_probs,
                                     self.discriminator_default_input_scale,
                                     self.discriminator_input_scales)
        y_hat0 = d.dropout_fprop(S, self.discriminator_default_input_include_prob,
                                     self.discriminator_input_include_probs,
                                     self.discriminator_default_input_scale,
                                     self.discriminator_input_scales)

        d_obj =  0.5 * (d.layers[-1].cost(y1, y_hat1) + d.layers[-1].cost(y0, y_hat0))

        if self.no_drop_in_d_for_g:
            y_hat0_no_drop = d.dropout_fprop(S)
            g_obj = d.layers[-1].cost(y1, y_hat0_no_drop)
        else:
            g_obj = d.layers[-1].cost(y1, y_hat0)

        if self.blend_obj:
            g_obj = (self.zurich_coeff * g_obj - self.minimax_coeff * d_obj) / (self.zurich_coeff + self.minimax_coeff)

        if model.inferer is not None:
            # Change this if we ever switch to using dropout in the
            # construction of S.
            S_nograd = block_gradient(S)  # Redundant as long as we have custom get_gradients
            pred = model.inferer.dropout_fprop(S_nograd, self.inference_default_input_include_prob,
                                                self.inference_input_include_probs,
                                                self.inference_default_input_scale,
                                                self.inference_input_scales)
            if self.infer_layer is None:
                target = z
            else:
                target = other_layers[self.infer_layer]
            i_obj = model.inferer.layers[-1].cost(target, pred)
        else:
            i_obj = 0

        return S, d_obj, g_obj, i_obj

    def get_gradients(self, model, data, **kwargs):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        assert isinstance(model, AdversaryPair)
        g = model.generator
        d = model.discriminator

        S, d_obj, g_obj, i_obj = self.get_samples_and_objectives(model, data)

        g_params = g.get_params()
        d_params = d.get_params()
        for param in g_params:
            assert param not in d_params
        for param in d_params:
            assert param not in g_params
        d_grads = T.grad(d_obj, d_params)
        g_grads = T.grad(g_obj, g_params)

        if self.scale_grads:
            S_grad = T.grad(g_obj, S)
            scale = T.maximum(1., self.target_scale / T.sqrt(T.sqr(S_grad).sum()))
            g_grads = [g_grad * scale for g_grad in g_grads]

        rval = OrderedDict()
        zeros = itertools.repeat(theano.tensor.constant(0., dtype='float32'))
        if self.ever_train_discriminator:
            rval.update(OrderedDict(safe_zip(d_params, [self.now_train_discriminator * dg for dg in d_grads])))
        else:
            rval.update(OrderedDict(zip(d_params, zeros)))
        if self.ever_train_generator:
            rval.update(OrderedDict(safe_zip(g_params, [self.now_train_generator * gg for gg in g_grads])))
        else:
            rval.update(OrderedDict(zip(g_params, zeros)))
        if self.ever_train_inference and model.inferer is not None:
            i_params = model.inferer.get_params()
            i_grads = T.grad(i_obj, i_params)
            rval.update(OrderedDict(safe_zip(i_params, [self.now_train_inference * ig for ig in i_grads])))
        elif model.inferer is not None:
            rval.update(OrderedDict(model.inferer.get_params(), zeros))

        updates = OrderedDict()

        # Two d steps for every g step
        if self.alternate_g:
            updates[self.now_train_generator] = 1. - self.now_train_generator

        return rval, updates

    def get_monitoring_channels(self, model, data, **kwargs):

        rval = OrderedDict()

        m = data.shape[0]

        g = model.generator
        d = model.discriminator

        y_hat = d.fprop(data)

        rval['false_negatives'] = T.cast((y_hat < 0.5).mean(), 'float32')

        samples = g.sample(m)
        y_hat = d.fprop(samples)
        rval['false_positives'] = T.cast((y_hat > 0.5).mean(), 'float32')
        # y = T.alloc(0., m, 1)
        cost = d.cost_from_X((samples, y_hat))
        sample_grad = T.grad(-cost, samples)
        rval['sample_grad_norm'] = T.sqrt(T.sqr(sample_grad).sum())
        _S, d_obj, g_obj, i_obj = self.get_samples_and_objectives(model, data)
        if model.monitor_inference and i_obj != 0:
            rval['objective_i'] = i_obj
        if model.monitor_discriminator:
            rval['objective_d'] = d_obj
        if model.monitor_generator:
            rval['objective_g'] = g_obj

        rval['now_train_generator'] = self.now_train_generator
        return rval

def recapitate_discriminator(pair_path, new_head):
    pair = serial.load(pair_path)
    d = pair.discriminator
    del d.layers[-1]
    d.add_layers([new_head])
    return d

def theano_parzen(data, mu, sigma):
    """
    Credit: Yann N. Dauphin
    """
    x = data

    a = ( x.dimshuffle(0, 'x', 1) - mu.dimshuffle('x', 0, 1) ) / sigma

    E = log_mean_exp(-0.5*(a**2).sum(2))

    Z = mu.shape[1] * T.log(sigma * numpy.sqrt(numpy.pi * 2))

    #return theano.function([x], E - Z)
    return E - Z


def log_mean_exp(a):
    """
    Credit: Yann N. Dauphin
    """

    max_ = a.max(1)

    return max_ + T.log(T.exp(a - max_.dimshuffle(0, 'x')).mean(1))

class Sum(Layer):
    """
    Monitoring channels are hardcoded for C01B batches
    """

    def __init__(self, layer_name):
        Model.__init__(self)
        self.__dict__.update(locals())
        del self.self
        self._params = []

    def set_input_space(self, space):
        self.input_space = space
        assert isinstance(space, CompositeSpace)
        self.output_space = space.components[0]

    def fprop(self, state_below):
        rval = state_below[0]
        for i in xrange(1, len(state_below)):
            rval = rval + state_below[i]
        rval.came_from_sum = True
        return rval

    @functools.wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                    state=None, targets=None):
        rval = OrderedDict()

        if state is None:
                state = self.fprop(state_below)
        vars_and_prefixes = [(state, '')]

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

        return rval

def marginals(dataset):
    return dataset.X.mean(axis=0)

class ActivateGenerator(TrainExtension):
    def __init__(self, active_after, value=1.):
        self.__dict__.update(locals())
        del self.self
        self.cur_epoch = 0

    def on_monitor(self, model, dataset, algorithm):
        if self.cur_epoch == self.active_after:
            algorithm.cost.now_train_generator.set_value(np.array(self.value, dtype='float32'))
        self.cur_epoch += 1

class InpaintingAdversaryCost(DefaultDataSpecsMixin, Cost):
    """
    """

    # Supplies own labels, don't get them from the dataset
    supervised = False

    def __init__(self, scale_grads=1, target_scale=.1,
            discriminator_default_input_include_prob = 1.,
            discriminator_input_include_probs=None,
            discriminator_default_input_scale=1.,
            discriminator_input_scales=None,
            generator_default_input_include_prob = 1.,
            generator_default_input_scale=1.,
            inference_default_input_include_prob=None,
            inference_input_include_probs=None,
            inference_default_input_scale=1.,
            inference_input_scales=None,
            init_now_train_generator=True,
            ever_train_discriminator=True,
            ever_train_generator=True,
            ever_train_inference=True,
            no_drop_in_d_for_g=False,
            alternate_g = False):
        self.__dict__.update(locals())
        del self.self
        # These allow you to dynamically switch off training parts.
        # If the corresponding ever_train_* is False, these have
        # no effect.
        self.now_train_generator = sharedX(init_now_train_generator)
        self.now_train_discriminator = sharedX(numpy.array(1., dtype='float32'))
        self.now_train_inference = sharedX(numpy.array(1., dtype='float32'))

    def expr(self, model, data, **kwargs):
        S, d_obj, g_obj, i_obj = self.get_samples_and_objectives(model, data)
        return d_obj + g_obj + i_obj

    def get_samples_and_objectives(self, model, data):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        assert isinstance(model, AdversaryPair)
        g = model.generator
        d = model.discriminator

        # Note: this assumes data is b01c
        X = data
        assert X.ndim == 4
        m = data.shape[space.get_batch_axis()]
        y1 = T.alloc(1, m, 1)
        y0 = T.alloc(0, m, 1)
        # NOTE: if this changes to optionally use dropout, change the inference
        # code below to use a non-dropped-out version.
        S, z = g.inpainting_sample_and_noise(X, default_input_include_prob=self.generator_default_input_include_prob, default_input_scale=self.generator_default_input_scale)
        y_hat1 = d.dropout_fprop(X, self.discriminator_default_input_include_prob,
                                     self.discriminator_input_include_probs,
                                     self.discriminator_default_input_scale,
                                     self.discriminator_input_scales)
        y_hat0 = d.dropout_fprop(S, self.discriminator_default_input_include_prob,
                                     self.discriminator_input_include_probs,
                                     self.discriminator_default_input_scale,
                                     self.discriminator_input_scales)

        d_obj =  0.5 * (d.layers[-1].cost(y1, y_hat1) + d.layers[-1].cost(y0, y_hat0))

        if self.no_drop_in_d_for_g:
            y_hat0_no_drop = d.dropout_fprop(S)
            g_obj = d.layers[-1].cost(y1, y_hat0)
        else:
            g_obj = d.layers[-1].cost(y1, y_hat0)

        if model.inferer is not None:
            # Change this if we ever switch to using dropout in the
            # construction of S.
            S_nograd = block_gradient(S)  # Redundant as long as we have custom get_gradients
            z_hat = model.inferer.dropout_fprop(S_nograd, self.inference_default_input_include_prob,
                                                self.inference_input_include_probs,
                                                self.inference_default_input_scale,
                                                self.inference_input_scales)
            i_obj = model.inferer.layers[-1].cost(z, z_hat)
        else:
            i_obj = 0

        return S, d_obj, g_obj, i_obj

    def get_gradients(self, model, data, **kwargs):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        assert isinstance(model, AdversaryPair)
        g = model.generator
        d = model.discriminator

        S, d_obj, g_obj, i_obj = self.get_samples_and_objectives(model, data)

        g_params = g.get_params()
        d_params = d.get_params()
        for param in g_params:
            assert param not in d_params
        for param in d_params:
            assert param not in g_params
        d_grads = T.grad(d_obj, d_params)
        g_grads = T.grad(g_obj, g_params)

        if self.scale_grads:
            S_grad = T.grad(g_obj, S)
            scale = T.maximum(1., self.target_scale / T.sqrt(T.sqr(S_grad).sum()))
            g_grads = [g_grad * scale for g_grad in g_grads]

        rval = OrderedDict()
        if self.ever_train_discriminator:
            rval.update(OrderedDict(safe_zip(d_params, [self.now_train_discriminator * dg for dg in d_grads])))
        else:
            rval.update(OrderedDict(zip(d_params, itertools.repeat(theano.tensor.constant(0., dtype='float32')))))

        if self.ever_train_generator:
            rval.update(OrderedDict(safe_zip(g_params, [self.now_train_generator * gg for gg in g_grads])))
        else:
            rval.update(OrderedDict(zip(g_params, itertools.repeat(theano.tensor.constant(0., dtype='float32')))))

        if self.ever_train_inference and model.inferer is not None:
            i_params = model.inferer.get_params()
            i_grads = T.grad(i_obj, i_params)
            rval.update(OrderedDict(safe_zip(i_params, [self.now_train_inference * ig for ig in i_grads])))

        updates = OrderedDict()

        # Two d steps for every g step
        if self.alternate_g:
            updates[self.now_train_generator] = 1. - self.now_train_generator

        return rval, updates

    def get_monitoring_channels(self, model, data, **kwargs):

        rval = OrderedDict()

        m = data.shape[0]

        g = model.generator
        d = model.discriminator

        y_hat = d.fprop(data)

        rval['false_negatives'] = T.cast((y_hat < 0.5).mean(), 'float32')

        samples, noise = g.inpainting_sample_and_noise(data)
        y_hat = d.fprop(samples)
        rval['false_positives'] = T.cast((y_hat > 0.5).mean(), 'float32')
        # y = T.alloc(0., m, 1)
        cost = d.cost_from_X((samples, y_hat))
        sample_grad = T.grad(-cost, samples)
        rval['sample_grad_norm'] = T.sqrt(T.sqr(sample_grad).sum())
        _S, d_obj, g_obj, i_obj = self.get_samples_and_objectives(model, data)
        if i_obj != 0:
            rval['objective_i'] = i_obj
        rval['objective_d'] = d_obj
        rval['objective_g'] = g_obj

        rval['now_train_generator'] = self.now_train_generator
        return rval

class Cycler(object):

    def __init__(self, k):
        self.__dict__.update(locals())
        del self.self
        self.i = 0

    def __call__(self, sgd):
        self.i = (self.i + 1) % self.k
        sgd.cost.now_train_generator.set_value(np.cast['float32'](self.i == 0))

class NoiseCat(Layer):

    def __init__(self, new_dim, std, layer_name):
        Layer.__init__(self)
        self.__dict__.update(locals())
        del self.self
        self._params = []

    def set_input_space(self, space):
        assert isinstance(space, VectorSpace)
        self.input_space = space
        self.output_space = VectorSpace(space.dim + self.new_dim)
        self.theano_rng = MRG_RandomStreams(self.mlp.rng.randint(2 ** 16))

    def fprop(self, state):
        noise = self.theano_rng.normal(std=self.std, avg=0., size=(state.shape[0], self.new_dim),
                dtype=state.dtype)
        return T.concatenate((state, noise), axis=1)

class RectifiedLinear(Layer):

    def __init__(self, layer_name, left_slope=0.0, **kwargs):
        super(RectifiedLinear, self).__init__(**kwargs)
        self.__dict__.update(locals())
        del self.self
        self._params = []

    def set_input_space(self, space):
        self.input_space = space
        self.output_space = space

    def fprop(self, state_below):
        p = state_below
        p = T.switch(p > 0., p, self.left_slope * p)
        return p

class Sigmoid(Layer):

    def __init__(self, layer_name, left_slope=0.0, **kwargs):
        super(Sigmoid, self).__init__(**kwargs)
        self.__dict__.update(locals())
        del self.self
        self._params = []

    def set_input_space(self, space):
        self.input_space = space
        self.output_space = space

    def fprop(self, state_below):
        p = T.nnet.sigmoid(state_below)
        return p

class SubtractHalf(Layer):

    def __init__(self, layer_name, left_slope=0.0, **kwargs):
        super(SubtractHalf, self).__init__(**kwargs)
        self.__dict__.update(locals())
        del self.self
        self._params = []

    def set_input_space(self, space):
        self.input_space = space
        self.output_space = space

    def fprop(self, state_below):
        return state_below - 0.5

    def get_weights(self):
        return self.mlp.layers[1].get_weights()

    def get_weights_format(self):
        return self.mlp.layers[1].get_weights_format()

    def get_weights_view_shape(self):
        return self.mlp.layers[1].get_weights_view_shape()

class SubtractRealMean(Layer):

    def __init__(self, layer_name, dataset, also_sd = False, **kwargs):
        super(SubtractRealMean, self).__init__(**kwargs)
        self.__dict__.update(locals())
        del self.self
        self._params = []
        self.mean = sharedX(dataset.X.mean(axis=0))
        if also_sd:
            self.sd = sharedX(dataset.X.std(axis=0))
        del self.dataset

    def set_input_space(self, space):
        self.input_space = space
        self.output_space = space

    def fprop(self, state_below):
        return (state_below - self.mean) / self.sd

    def get_weights(self):
        return self.mlp.layers[1].get_weights()

    def get_weights_format(self):
        return self.mlp.layers[1].get_weights_format()

    def get_weights_view_shape(self):
        return self.mlp.layers[1].get_weights_view_shape()


class Clusterize(Layer):

    def __init__(self, scale, layer_name):
        Layer.__init__(self)
        self.__dict__.update(locals())
        del self.self
        self._params = []

    def set_input_space(self, space):
        assert isinstance(space, VectorSpace)
        self.input_space = space
        self.output_space = space
        self.theano_rng = MRG_RandomStreams(self.mlp.rng.randint(2 ** 16))

    def fprop(self, state):
        noise = self.theano_rng.binomial(size=state.shape, p=0.5,
                dtype=state.dtype) * 2. - 1.
        return state + self.scale * noise



class ThresholdedAdversaryCost(DefaultDataSpecsMixin, Cost):
    """
    """

    # Supplies own labels, don't get them from the dataset
    supervised = False

    def __init__(self, scale_grads=1, target_scale=.1,
            discriminator_default_input_include_prob = 1.,
            discriminator_input_include_probs=None,
            discriminator_default_input_scale=1.,
            discriminator_input_scales=None,
            generator_default_input_include_prob = 1.,
            generator_default_input_scale=1.,
            inference_default_input_include_prob=None,
            inference_input_include_probs=None,
            inference_default_input_scale=1.,
            inference_input_scales=None,
            init_now_train_generator=True,
            ever_train_discriminator=True,
            ever_train_generator=True,
            ever_train_inference=True,
            no_drop_in_d_for_g=False,
            alternate_g = False,
            infer_layer=None,
            noise_both = 0.):
        self.__dict__.update(locals())
        del self.self
        # These allow you to dynamically switch off training parts.
        # If the corresponding ever_train_* is False, these have
        # no effect.
        self.now_train_generator = sharedX(init_now_train_generator)
        self.now_train_discriminator = sharedX(numpy.array(1., dtype='float32'))
        self.now_train_inference = sharedX(numpy.array(1., dtype='float32'))

    def expr(self, model, data, **kwargs):
        S, d_obj, g_obj, i_obj = self.get_samples_and_objectives(model, data)
        l = []
        # This stops stuff from ever getting computed if we're not training
        # it.
        if self.ever_train_discriminator:
            l.append(d_obj)
        if self.ever_train_generator:
            l.append(g_obj)
        if self.ever_train_inference:
            l.append(i_obj)
        return sum(l)

    def get_samples_and_objectives(self, model, data):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        assert isinstance(model, AdversaryPair)
        g = model.generator
        d = model.discriminator

        # Note: this assumes data is design matrix
        X = data
        m = data.shape[space.get_batch_axis()]
        y1 = T.alloc(1, m, 1)
        y0 = T.alloc(0, m, 1)
        # NOTE: if this changes to optionally use dropout, change the inference
        # code below to use a non-dropped-out version.
        S, z, other_layers = g.sample_and_noise(m, default_input_include_prob=self.generator_default_input_include_prob, default_input_scale=self.generator_default_input_scale, all_g_layers=(self.infer_layer is not None))

        if self.noise_both != 0.:
            rng = MRG_RandomStreams(2014 / 6 + 2)
            S = S + rng.normal(size=S.shape, dtype=S.dtype) * self.noise_both
            X = X + rng.normal(size=X.shape, dtype=S.dtype) * self.noise_both

        y_hat1 = d.dropout_fprop(X, self.discriminator_default_input_include_prob,
                                     self.discriminator_input_include_probs,
                                     self.discriminator_default_input_scale,
                                     self.discriminator_input_scales)
        y_hat0 = d.dropout_fprop(S, self.discriminator_default_input_include_prob,
                                     self.discriminator_input_include_probs,
                                     self.discriminator_default_input_scale,
                                     self.discriminator_input_scales)

        d_obj =  0.5 * (d.layers[-1].cost(y1, y_hat1) + d.layers[-1].cost(y0, y_hat0))

        if self.no_drop_in_d_for_g:
            y_hat0_no_drop = d.dropout_fprop(S)
            g_cost_mat = d.layers[-1].cost_matrix(y1, y_hat0_no_drop)
        else:
            g_cost_mat = d.layers[-1].cost_matrix(y1, y_hat0)
        assert g_cost_mat.ndim == 2
        assert y_hat0.ndim == 2

        mask = y_hat0 < 0.5
        masked_cost = g_cost_mat * mask
        g_obj = masked_cost.mean()


        if model.inferer is not None:
            # Change this if we ever switch to using dropout in the
            # construction of S.
            S_nograd = block_gradient(S)  # Redundant as long as we have custom get_gradients
            pred = model.inferer.dropout_fprop(S_nograd, self.inference_default_input_include_prob,
                                                self.inference_input_include_probs,
                                                self.inference_default_input_scale,
                                                self.inference_input_scales)
            if self.infer_layer is None:
                target = z
            else:
                target = other_layers[self.infer_layer]
            i_obj = model.inferer.layers[-1].cost(target, pred)
        else:
            i_obj = 0

        return S, d_obj, g_obj, i_obj

    def get_gradients(self, model, data, **kwargs):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        assert isinstance(model, AdversaryPair)
        g = model.generator
        d = model.discriminator

        S, d_obj, g_obj, i_obj = self.get_samples_and_objectives(model, data)

        g_params = g.get_params()
        d_params = d.get_params()
        for param in g_params:
            assert param not in d_params
        for param in d_params:
            assert param not in g_params
        d_grads = T.grad(d_obj, d_params)
        g_grads = T.grad(g_obj, g_params)

        if self.scale_grads:
            S_grad = T.grad(g_obj, S)
            scale = T.maximum(1., self.target_scale / T.sqrt(T.sqr(S_grad).sum()))
            g_grads = [g_grad * scale for g_grad in g_grads]

        rval = OrderedDict()
        zeros = itertools.repeat(theano.tensor.constant(0., dtype='float32'))
        if self.ever_train_discriminator:
            rval.update(OrderedDict(safe_zip(d_params, [self.now_train_discriminator * dg for dg in d_grads])))
        else:
            rval.update(OrderedDict(zip(d_params, zeros)))
        if self.ever_train_generator:
            rval.update(OrderedDict(safe_zip(g_params, [self.now_train_generator * gg for gg in g_grads])))
        else:
            rval.update(OrderedDict(zip(g_params, zeros)))
        if self.ever_train_inference and model.inferer is not None:
            i_params = model.inferer.get_params()
            i_grads = T.grad(i_obj, i_params)
            rval.update(OrderedDict(safe_zip(i_params, [self.now_train_inference * ig for ig in i_grads])))
        elif model.inferer is not None:
            rval.update(OrderedDict(model.inferer.get_params(), zeros))

        updates = OrderedDict()

        # Two d steps for every g step
        if self.alternate_g:
            updates[self.now_train_generator] = 1. - self.now_train_generator

        return rval, updates

    def get_monitoring_channels(self, model, data, **kwargs):

        rval = OrderedDict()

        m = data.shape[0]

        g = model.generator
        d = model.discriminator

        y_hat = d.fprop(data)

        rval['false_negatives'] = T.cast((y_hat < 0.5).mean(), 'float32')

        samples = g.sample(m)
        y_hat = d.fprop(samples)
        rval['false_positives'] = T.cast((y_hat > 0.5).mean(), 'float32')
        # y = T.alloc(0., m, 1)
        cost = d.cost_from_X((samples, y_hat))
        sample_grad = T.grad(-cost, samples)
        rval['sample_grad_norm'] = T.sqrt(T.sqr(sample_grad).sum())
        _S, d_obj, g_obj, i_obj = self.get_samples_and_objectives(model, data)
        if model.monitor_inference and i_obj != 0:
            rval['objective_i'] = i_obj
        if model.monitor_discriminator:
            rval['objective_d'] = d_obj
        if model.monitor_generator:
            rval['objective_g'] = g_obj

        rval['now_train_generator'] = self.now_train_generator
        return rval


class HardSigmoid(Linear):
    """
    Hard "sigmoid" (note: shifted along the x axis)
    """

    def __init__(self, left_slope=0.0, **kwargs):
        super(HardSigmoid, self).__init__(**kwargs)
        self.left_slope = left_slope

    @wraps(Layer.fprop)
    def fprop(self, state_below):

        p = self._linear_part(state_below)
        # Original: p = p * (p > 0.) + self.left_slope * p * (p < 0.)
        # T.switch is faster.
        # For details, see benchmarks in
        # pylearn2/scripts/benchmark/time_relu.py
        p = T.clip(p, 0., 1.)
        return p

    @wraps(Layer.cost)
    def cost(self, *args, **kwargs):

        raise NotImplementedError()


class LazyAdversaryCost(DefaultDataSpecsMixin, Cost):
    """
    """

    # Supplies own labels, don't get them from the dataset
    supervised = False

    def __init__(self, scale_grads=1, target_scale=.1,
            discriminator_default_input_include_prob = 1.,
            discriminator_input_include_probs=None,
            discriminator_default_input_scale=1.,
            discriminator_input_scales=None,
            generator_default_input_include_prob = 1.,
            generator_default_input_scale=1.,
            inference_default_input_include_prob=None,
            inference_input_include_probs=None,
            inference_default_input_scale=1.,
            inference_input_scales=None,
            init_now_train_generator=True,
            ever_train_discriminator=True,
            ever_train_generator=True,
            ever_train_inference=True,
            no_drop_in_d_for_g=False,
            alternate_g = False,
            infer_layer=None,
            noise_both = 0.,
            g_eps = 0.,
            d_eps =0.):
        self.__dict__.update(locals())
        del self.self
        # These allow you to dynamically switch off training parts.
        # If the corresponding ever_train_* is False, these have
        # no effect.
        self.now_train_generator = sharedX(init_now_train_generator)
        self.now_train_discriminator = sharedX(numpy.array(1., dtype='float32'))
        self.now_train_inference = sharedX(numpy.array(1., dtype='float32'))

    def expr(self, model, data, **kwargs):
        S, d_obj, g_obj, i_obj = self.get_samples_and_objectives(model, data)
        l = []
        # This stops stuff from ever getting computed if we're not training
        # it.
        if self.ever_train_discriminator:
            l.append(d_obj)
        if self.ever_train_generator:
            l.append(g_obj)
        if self.ever_train_inference:
            l.append(i_obj)
        return sum(l)

    def get_samples_and_objectives(self, model, data):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        assert isinstance(model, AdversaryPair)
        g = model.generator
        d = model.discriminator

        # Note: this assumes data is design matrix
        X = data
        m = data.shape[space.get_batch_axis()]
        y1 = T.alloc(1, m, 1)
        y0 = T.alloc(0, m, 1)
        # NOTE: if this changes to optionally use dropout, change the inference
        # code below to use a non-dropped-out version.
        S, z, other_layers = g.sample_and_noise(m, default_input_include_prob=self.generator_default_input_include_prob, default_input_scale=self.generator_default_input_scale, all_g_layers=(self.infer_layer is not None))

        if self.noise_both != 0.:
            rng = MRG_RandomStreams(2014 / 6 + 2)
            S = S + rng.normal(size=S.shape, dtype=S.dtype) * self.noise_both
            X = X + rng.normal(size=X.shape, dtype=S.dtype) * self.noise_both

        y_hat1 = d.dropout_fprop(X, self.discriminator_default_input_include_prob,
                                     self.discriminator_input_include_probs,
                                     self.discriminator_default_input_scale,
                                     self.discriminator_input_scales)
        y_hat0 = d.dropout_fprop(S, self.discriminator_default_input_include_prob,
                                     self.discriminator_input_include_probs,
                                     self.discriminator_default_input_scale,
                                     self.discriminator_input_scales)

        # d_obj =  0.5 * (d.layers[-1].cost(y1, y_hat1) + d.layers[-1].cost(y0, y_hat0))

        pos_mask = y_hat1 < .5 + self.d_eps
        neg_mask = y_hat0 > .5 - self.d_eps

        pos_cost_matrix = d.layers[-1].cost_matrix(y1, y_hat1)
        neg_cost_matrix = d.layers[-1].cost_matrix(y0, y_hat0)

        pos_cost = (pos_mask * pos_cost_matrix).mean()
        neg_cost = (neg_mask * neg_cost_matrix).mean()

        d_obj = 0.5 * (pos_cost + neg_cost)

        if self.no_drop_in_d_for_g:
            y_hat0_no_drop = d.dropout_fprop(S)
            g_cost_mat = d.layers[-1].cost_matrix(y1, y_hat0_no_drop)
        else:
            g_cost_mat = d.layers[-1].cost_matrix(y1, y_hat0)
        assert g_cost_mat.ndim == 2
        assert y_hat0.ndim == 2

        mask = y_hat0 < 0.5 + self.g_eps
        masked_cost = g_cost_mat * mask
        g_obj = masked_cost.mean()


        if model.inferer is not None:
            # Change this if we ever switch to using dropout in the
            # construction of S.
            S_nograd = block_gradient(S)  # Redundant as long as we have custom get_gradients
            pred = model.inferer.dropout_fprop(S_nograd, self.inference_default_input_include_prob,
                                                self.inference_input_include_probs,
                                                self.inference_default_input_scale,
                                                self.inference_input_scales)
            if self.infer_layer is None:
                target = z
            else:
                target = other_layers[self.infer_layer]
            i_obj = model.inferer.layers[-1].cost(target, pred)
        else:
            i_obj = 0

        return S, d_obj, g_obj, i_obj

    def get_gradients(self, model, data, **kwargs):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        assert isinstance(model, AdversaryPair)
        g = model.generator
        d = model.discriminator

        S, d_obj, g_obj, i_obj = self.get_samples_and_objectives(model, data)

        g_params = g.get_params()
        d_params = d.get_params()
        for param in g_params:
            assert param not in d_params
        for param in d_params:
            assert param not in g_params
        d_grads = T.grad(d_obj, d_params)
        g_grads = T.grad(g_obj, g_params)

        if self.scale_grads:
            S_grad = T.grad(g_obj, S)
            scale = T.maximum(1., self.target_scale / T.sqrt(T.sqr(S_grad).sum()))
            g_grads = [g_grad * scale for g_grad in g_grads]

        rval = OrderedDict()
        zeros = itertools.repeat(theano.tensor.constant(0., dtype='float32'))
        if self.ever_train_discriminator:
            rval.update(OrderedDict(safe_zip(d_params, [self.now_train_discriminator * dg for dg in d_grads])))
        else:
            rval.update(OrderedDict(zip(d_params, zeros)))
        if self.ever_train_generator:
            rval.update(OrderedDict(safe_zip(g_params, [self.now_train_generator * gg for gg in g_grads])))
        else:
            rval.update(OrderedDict(zip(g_params, zeros)))
        if self.ever_train_inference and model.inferer is not None:
            i_params = model.inferer.get_params()
            i_grads = T.grad(i_obj, i_params)
            rval.update(OrderedDict(safe_zip(i_params, [self.now_train_inference * ig for ig in i_grads])))
        elif model.inferer is not None:
            rval.update(OrderedDict(model.inferer.get_params(), zeros))

        updates = OrderedDict()

        # Two d steps for every g step
        if self.alternate_g:
            updates[self.now_train_generator] = 1. - self.now_train_generator

        return rval, updates

    def get_monitoring_channels(self, model, data, **kwargs):

        rval = OrderedDict()

        m = data.shape[0]

        g = model.generator
        d = model.discriminator

        y_hat = d.fprop(data)

        rval['false_negatives'] = T.cast((y_hat < 0.5).mean(), 'float32')

        samples = g.sample(m)
        y_hat = d.fprop(samples)
        rval['false_positives'] = T.cast((y_hat > 0.5).mean(), 'float32')
        # y = T.alloc(0., m, 1)
        cost = d.cost_from_X((samples, y_hat))
        sample_grad = T.grad(-cost, samples)
        rval['sample_grad_norm'] = T.sqrt(T.sqr(sample_grad).sum())
        _S, d_obj, g_obj, i_obj = self.get_samples_and_objectives(model, data)
        if model.monitor_inference and i_obj != 0:
            rval['objective_i'] = i_obj
        if model.monitor_discriminator:
            rval['objective_d'] = d_obj
        if model.monitor_generator:
            rval['objective_g'] = g_obj

        rval['now_train_generator'] = self.now_train_generator
        return rval
