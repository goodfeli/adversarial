from pylearn2.utils import serial
import sys
_, model_path = sys.argv
model = serial.load(model_path)
from pylearn2.gui.patch_viewer import make_viewer
space = model.generator.get_output_space()
from pylearn2.config import yaml_parse
import numpy as np

dataset = yaml_parse.load(model.dataset_yaml_src)
dataset = dataset.get_test_set()

grid_shape = None

from pylearn2.utils import sharedX
X = sharedX(dataset.get_batch_topo(100))
samples, ignore = model.generator.inpainting_sample_and_noise(X)
samples = samples.eval()
total_dimension = space.get_total_dimension()
num_colors = 1
if total_dimension % 3 == 0:
    num_colors = 3
w = int(np.sqrt(total_dimension / num_colors))
from pylearn2.space import Conv2DSpace
desired_space = Conv2DSpace(shape=[w, w], num_channels=num_colors, axes=('b',0,1,'c'))
is_color = samples.shape[-1] == 3
print (samples.min(), samples.mean(), samples.max())
# Hack for detecting MNIST [0, 1] values. Otherwise we assume centered images
if samples.min() >0:
    samples = samples * 2.0 - 1.0
viewer = make_viewer(samples, grid_shape=grid_shape, is_color=is_color)
viewer.show()
