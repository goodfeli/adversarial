import numpy as np
import sys

from theano import function
from theano import tensor as T

_, model_path, sigma = sys.argv
from pylearn2.utils import serial
model = serial.load(model_path)
from pylearn2.config import yaml_parse
dataset = yaml_parse.load(model.dataset_yaml_src)
dataset = dataset.get_test_set()
from pylearn2.utils import sharedX
g = model.generator
n = g.get_input_space().get_total_dimension()
X = sharedX(dataset.X)
from theano.sandbox.rng_mrg import MRG_RandomStreams
theano_rng = MRG_RandomStreams(2014 + 6 * 24)
assert False # Aaron says to do valid comparison we need to download the standard binarization,
# and the model should also have been trained on the standard binarization
f = function([], updates=[(X, theano_rng.binomial(p=X, size=X.shape, dtype=X.dtype))])
f()
m = dataset.X.shape[0]
accumulator = sharedX(np.zeros((m,)))
z_samples = g.get_noise(1)
x_samples = g.mlp.fprop(z_samples)
# x_samples = X
from theano.compat import OrderedDict
updates = OrderedDict()
from theano import shared
num_samples = shared(1)
sigma = sharedX(float(sigma))
prev = accumulator
from theano.printing import Print
#prev = Print('prev',attrs=['min','max'])(prev)
# E_x log E_z exp(- sum_i softplus( (1 - 2 x_i) A(z)_i) )
from pylearn2.expr.nnet import arg_of_sigmoid
A = arg_of_sigmoid(x_samples)
cur = - T.nnet.softplus((1. - 2. * X) * A).sum(axis=1)
#cur = Print('cur',attrs=['min','max'])(cur)
ofs = T.maximum(prev, cur)
num_samples_f = T.cast(num_samples, 'float32')
updates[accumulator] = ofs + T.log((num_samples_f * T.exp(prev - ofs) + T.exp(cur - ofs)) / (num_samples_f + 1.))
updates[num_samples] = num_samples + 1
f = function([], updates=updates)
updates[accumulator] = cur
del updates[num_samples]
first = function([], updates=updates)
avg_ll = accumulator.mean()

import time
prev_t = time.time()
first()
while True:
    v = avg_ll.eval()
    i = num_samples.get_value()
    if i == 1 or i % 1000 == 0:
        now_t = time.time()
        print i, v, now_t - prev_t
        prev_t = now_t
    if np.isnan(v) or np.isinf(v):
        break
    f()

# E_x log p(x)
# E_x log int p(x, z) dz
# E_x log int p(z) p(x | z) dz
# E_x log E_z p(x | z)
# E_x log E_z prod_i p(x_i | z)
# E_x log E_z prod_i sigmoid( (2 x_i - 1) A(z)_i)
# E_x log E_z exp(log prod_i sigmoid( (2 x_i - 1) A(z)_i) )
# E_x log E_z exp(sum_i log sigmoid( (2 x_i - 1) A(z)_i) )
# E_x log E_z exp(- sum_i softplus( (1 - 2 x_i) A(z)_i) )
