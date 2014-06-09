import theano
from pylearn2.utils import serial
import sys
from pylearn2.gui.patch_viewer import make_viewer
from pylearn2.space import VectorSpace
from pylearn2.config import yaml_parse
import numpy as np
import ipdb


# TODO, only works for CIFAR10 for now

grid_shape = None
repeat_samples = 1
num_samples = 5


_, model_path = sys.argv
model = serial.load(model_path)
rng = np.random.RandomState(20232)

def get_data_samples(dataset, n = num_samples):
    unique_y = np.unique(dataset.y)
    rval = []
    for y in np.unique(dataset.y):
        ind = np.where(dataset.y == y)[0]
        ind = ind[rng.randint(0, len(ind), n)]
        rval.append(dataset.get_topological_view()[ind])

    return np.concatenate(rval)

dataset = yaml_parse.load(model.dataset_yaml_src)
dataset = dataset.get_test_set()
data = get_data_samples(dataset)

output_space = model.generator.get_output_space()
input_space = model.generator.mlp.input_space

X = input_space.get_theano_batch()
samples, _ = model.generator.inpainting_sample_and_noise(X)
f = theano.function([X], samples)

samples = []
for i in xrange(repeat_samples):
    samples.append(f(data))

samples = np.concatenate(samples)

is_color = True


print (samples.min(), samples.mean(), samples.max())
# Hack for detecting MNIST [0, 1] values. Otherwise we assume centered images
if samples.min() >0:
    samples = samples * 2.0 - 1.0
viewer = make_viewer(samples, grid_shape=grid_shape, is_color=is_color)
viewer.show()
