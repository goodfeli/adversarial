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
m = dataset.X.shape[0]
accumulator = sharedX(np.zeros((m,)))
z_samples = g.get_noise(1)
x_samples = g.mlp.fprop(z_samples)
from theano.compat import OrderedDict
updates = OrderedDict()
from theano import shared
num_samples = shared(1)
sigma = sharedX(float(sigma))
prev = accumulator
cur = -0.5 * T.sqr(X - x_samples).sum(axis=1) / T.sqr(sigma)
ofs = T.maximum(prev, cur)
num_samples_f = T.cast(num_samples, 'float32')
updates[accumulator] = ofs + T.log(num_samples_f * T.exp(prev - ofs) + T.exp(cur - ofs)) - T.log(num_samples_f + 1.)
updates[num_samples] = num_samples + 1
f = function([], updates=updates)
updates[accumulator] = cur
del updates[num_samples]
first = function([], updates=updates)
avg_ll = accumulator.mean() - 0.5 * X.shape[1] * T.log(2 * np.pi * T.sqr(sigma))

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

# log p(x)
# = log int p(z, x) dz
# = log int p(z) p(x |z) dz
# = log E_z p(x|z)
# = log (1/m) sum_z p(x|z)
# = log (1/m) sum_z prod_i sqrt(1/(2 pi sigma^2)) exp( -0.5 (x_i-g(z)_i)^2 / sigma^2)
# = log  sqrt(1/(2 pi sigma^2))^d (1/m) sum_z prod_iexp( -0.5 (x_i-g(z)_i)^2 / sigma^2)
# = log  sqrt(1/(2 pi sigma^2))^d (1/m) sum_z exp( sum_i -0.5 (x_i-g(z)_i)^2 / sigma^2)
# = log  sqrt(1/(2 pi sigma^2))^d + log (1/m) sum_z exp( sum_i -0.5 (x_i-g(z)_i)^2 / sigma^2)
# = 0.5 d log  1/(2 pi sigma^2) + log (1/m) sum_z exp( sum_i -0.5 (x_i-g(z)_i)^2 / sigma^2)
# = -0.5 d log  (2 pi sigma^2) + log (1/m) sum_z exp( sum_i -0.5 (x_i-g(z)_i)^2 / sigma^2)

# log (1/m) sum_j exp(v_j)
# = log (1/m) [exp(v_m) + sum_{j=1}^{m-1} exp(v_j)]
# = log (1/m) [exp(v_m) + (m-1) exp( prev )]
# = log (1/m) [exp(v_m) exp(ofs-ofs) + (m-1) exp( prev ) exp(ofs -ofs)]
# = log (1/m) [exp(v_m- ofs) exp(ofs) + (m-1) exp( prev -ofs) exp(ofs)]
# = log exp(ofs) (1/m) [exp(v_m- ofs) + (m-1) exp( prev -ofs) ]
# = ofs + log  (1/m) [exp(v_m- ofs) + (m-1) exp( prev -ofs) ]
# = ofs + log  [exp(v_m- ofs) + (m-1) exp( prev -ofs) ] - log m
