from pylearn2.utils import serial
import sys
_, model_path = sys.argv
model = serial.load(model_path)
from pylearn2.gui.patch_viewer import make_viewer
space = model.generator.get_output_space()
from pylearn2.config import yaml_parse
from pylearn2.gui.patch_viewer import PatchViewer
import numpy as np

dataset = yaml_parse.load(model.dataset_yaml_src)

grid_shape = None

rows = 4
sample_cols = 5

# For some reason format_as from VectorSpace is not working right
samples = model.generator.sample(rows * sample_cols).eval()
topo_samples = dataset.get_topological_view(samples)

pv = PatchViewer(grid_shape=(rows, sample_cols + 1), patch_shape=(28,28),
        is_color=False)

X = dataset.X
topo = dataset.get_topological_view()
index = 0
for i in xrange(samples.shape[0]):
    topo_sample = topo_samples[i, :, :, :]
    pv.add_patch(topo_sample * 2. - 1., rescale=False)

    if (i +1) % sample_cols == 0:
        sample = samples[i, :]
        dists = np.square(X - sample).sum(axis=1)
        j = np.argmin(dists)
        match = topo[j, :]
        pv.add_patch(match * 2 -1, rescale=False, activation=1)

pv.show()
