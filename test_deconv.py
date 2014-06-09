"""
This script visually test the deconv layer.
Construct an MLP with conv ,and deconv layer,
set their W to same values and show the original
input and the output of the mlp side by side.
They are supposed to look same.
"""


import theano
from adversarial.deconv import Deconv
from pylearn2.datasets.mnist import MNIST
from pylearn2.space import Conv2DSpace
from pylearn2.models.mlp import MLP
from pylearn2.models.maxout import MaxoutConvC01B
from pylearn2.gui import patch_viewer
import ipdb


input_space = Conv2DSpace(shape = (28, 28), num_channels=1, axes = ('c', 0, 1, 'b'))
conv = MaxoutConvC01B(layer_name = 'conv',
                        num_channels = 16,
                        num_pieces = 1,
                        kernel_shape = (4, 4),
                        pool_shape = (1, 1),
                        pool_stride=(1, 1),
                        irange = 0.05)
deconv = Deconv(layer_name = 'deconv',
                num_channels = 1,
                kernel_shape = (4, 4),
                irange = 0.05)

mlp = MLP(input_space =input_space,
        layers = [conv, deconv])

mlp.layers[1].transformer._filters.set_value(mlp.layers[0].transformer._filters.get_value())

x = input_space.get_theano_batch()
out = mlp.fprop(x)
f = theano.function([x], out)

data = MNIST('test')
data_specs = (input_space, 'features')
iter = data.iterator(mode = 'sequential', batch_size = 2, data_specs = data_specs)
pv = patch_viewer.PatchViewer((10, 10), (28, 28))
for item in iter:
    res = f(item)
    pv.add_patch(item[0,:,:,0])
    pv.add_patch(res[0,:,:,0])
    pv.show()
    break

