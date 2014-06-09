from pylearn2.utils import serial
import sys
_, model_path = sys.argv
model = serial.load(model_path)
from pylearn2.gui.patch_viewer import make_viewer
space = model.generator.get_output_space()
from pylearn2.space import VectorSpace
from pylearn2.config import yaml_parse
import numpy as np

match_train = True
if match_train:
    dataset = yaml_parse.load(model.dataset_yaml_src)

grid_shape = None

if isinstance(space, VectorSpace):
    # For some reason format_as from VectorSpace is not working right
    samples = model.generator.sample(100).eval()

    if match_train:
        grid_shape = (10, 20)
        matched = np.zeros((samples.shape[0] * 2, samples.shape[1]))
        X = dataset.X
        for i in xrange(samples.shape[0]):
            matched[2 * i, :] = samples[i, :].copy()
            dists = np.square(X - samples[i, :]).sum(axis=1)
            j = np.argmin(dists)
            matched[2 * i + 1, :] = X[j, :]
        samples = matched

    is_color = samples.shape[-1] % 3 == 0 and samples.shape[-1] != 48 * 48
else:
    total_dimension = space.get_total_dimension()
    import numpy as np
    num_colors = 1
    if total_dimension % 3 == 0:
        num_colors = 3
    w = int(np.sqrt(total_dimension / num_colors))
    from pylearn2.space import Conv2DSpace
    desired_space = Conv2DSpace(shape=[w, w], num_channels=num_colors, axes=('b',0,1,'c'))
    samples = space.format_as(batch=model.generator.sample(100),
            space=desired_space).eval()
    is_color = samples.shape[-1] == 3
print (samples.min(), samples.mean(), samples.max())
# Hack for detecting MNIST [0, 1] values. Otherwise we assume centered images
if samples.min() >0:
    samples = samples * 2.0 - 1.0
viewer = make_viewer(samples, grid_shape=grid_shape, is_color=is_color)
viewer.show()
