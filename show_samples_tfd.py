from pylearn2.utils import serial
import sys
_, model_path = sys.argv
model = serial.load(model_path)
from pylearn2.gui.patch_viewer import make_viewer
space = model.generator.get_output_space()
total_dimension = space.get_total_dimension()
import numpy as np
num_colors = 1
#if total_dimension % 3 == 0:
#    num_colors = 3
w = int(np.sqrt(total_dimension / num_colors))
from pylearn2.space import Conv2DSpace
desired_space = Conv2DSpace(shape=[w, w], num_channels=num_colors, axes=('b',0,1,'c'))
samples = space.format_as(batch=model.generator.sample(100),
        space=desired_space).eval()
print (samples.min(), samples.mean(), samples.max())
viewer = make_viewer(samples * 2.0 - 1.0)
viewer.show()
