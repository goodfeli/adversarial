Generative Adversarial Networks
===============================

This repository contains the code and hyperparameters for the paper:

"Generative Adversarial Networks." Ian J. Goodfellow, Jean Pouget-Abadie,
Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville,
Yoshua Bengio. ArXiv 2014.

Please cite this paper if you use the code in this repository as part of
a published research project.

We are an academic lab, not a software company, and have no personnel
devoted to documenting and maintaing this research code.
Therefore this code is offered with absolutely no support.
Exact reproduction of the numbers in the paper depends on exact
reproduction of many factors,
including the version of all software dependencies and the choice of
underlying hardware (GPU model, etc). We used NVIDA Ge-Force GTX-580
graphics cards; other hardware will use different tree structures for
summation and incur different rounding error. If you do not reproduce our
setup exactly you should expect to need to re-tune your hyperparameters
slight for your new setup.

Moreover, we have not integrated any unit tests for this code into Theano
or Pylearn2 so subsequent changes to those libraries may break the code
in this repository. If you encounter problems with this code, you should
make sure that you are using the development branch of Pylearn2 and Theano,
and use "git checkout" to go to a commit from approximately June 9, 2014.

This code itself requires no installation besides making sure that the
"adversarial" directory is in a directory in your PYTHONPATH. If
installed correctly, 'python -c "import adversarial"' will work. You
must also install Pylearn2 and Pylearn2's dependencies (Theano, numpy,
etc.)

parzen_ll.py is the script used to estimate the log likelihood of the
model using the Parzen density technique.

Call pylearn2/scripts/train.py on the various yaml files in this repository
to train the model for each dataset reported in the paper. The names of
*.yaml are fairly self-explanatory.
