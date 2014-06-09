import sys
from pylearn2.gui.patch_viewer import make_viewer
from pylearn2.utils import serial
model = serial.load(sys.argv[1])
generator = model.generator

final = generator.mlp.layers[-1]
success = False

i = -1
success = False
to_search = generator.mlp
while not success:
    print "while loop ", i
    final = to_search.layers[i]
    if 'Composite' in str(type(final)):
        i = input("which")
        elem = final.layers[i]
        if hasattr(elem, 'layers'):
            print "stepping into inner MLP"
            i = -1
            to_search = elem
            continue
        else:
            print "examining this element"
            final = elem

    try:
        print "Trying get_weights topo"
        topo = final.get_weights_topo()
        print "It worked"
        success = True
    except Exception:
        pass

    if success:
        print "Making the viewer and showing"
        make_viewer(topo).show()
        quit()

    try:
        print "Trying get_weights"
        weights = final.get_weights()
        print "It worked"
        success = True
    except NotImplementedError:
        i -= 1 # skip over SpaceConverter, etc.
print "Out of the while loop"


print "weights shape ", weights.shape
viewer = make_viewer(weights, is_color=weights.shape[1] % 3 == 0 and weights.shape[1] != 48*48)
print "image shape ", viewer.image.shape

print "made viewer"

viewer.show()

print "executed show"
