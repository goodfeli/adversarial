import argparse
import time
import gc
import numpy
import theano
import theano.tensor as T
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from pylearn2.datasets.mnist import MNIST
from pylearn2.datasets.tfd import TFD



def get_nll(x, parzen, batch_size=10):
    """
    Credit: Yann N. Dauphin
    """

    inds = range(x.shape[0])
    n_batches = int(numpy.ceil(float(len(inds)) / batch_size))

    times = []
    nlls = []
    for i in range(n_batches):
        begin = time.time()
        nll = parzen(x[inds[i::n_batches]])
        end = time.time()
        times.append(end-begin)
        nlls.extend(nll)

        if i % 10 == 0:
            print i, numpy.mean(times), numpy.mean(nlls)

    return numpy.array(nlls)


def log_mean_exp(a):
    """
    Credit: Yann N. Dauphin
    """

    max_ = a.max(1)

    return max_ + T.log(T.exp(a - max_.dimshuffle(0, 'x')).mean(1))


def theano_parzen(mu, sigma):
    """
    Credit: Yann N. Dauphin
    """

    x = T.matrix()
    mu = theano.shared(mu)
    a = ( x.dimshuffle(0, 'x', 1) - mu.dimshuffle('x', 0, 1) ) / sigma
    E = log_mean_exp(-0.5*(a**2).sum(2))
    Z = mu.shape[1] * T.log(sigma * numpy.sqrt(numpy.pi * 2))

    return theano.function([x], E - Z)


def cross_validate_sigma(samples, data, sigmas, batch_size):

    lls = []
    for sigma in sigmas:
        print sigma
        parzen = theano_parzen(samples, sigma)
        tmp = get_nll(data, parzen, batch_size = batch_size)
        lls.append(numpy.asarray(tmp).mean())
        del parzen
        gc.collect()

    ind = numpy.argmax(lls)
    return sigmas[ind]


def get_valid(ds, limit_size = -1, fold = 0):
    if ds == 'mnist':
        data = MNIST('train', start=50000, stop=60000)
        return data.X[:limit_size]
    elif ds == 'tfd':
        data = TFD('valid', fold = fold, scale=True)
        return data.X
    else:
         raise ValueError("Unknow dataset: {}".format(args.dataet))


def get_test(ds, test, fold=0):
    if ds == 'mnist':
        return test.get_test_set()
    elif ds == 'tfd':
        return test.get_test_set(fold=fold)
    else:
        raise ValueError("Unknow dataset: {}".format(args.dataet))


def main():
    parser = argparse.ArgumentParser(description = 'Parzen window, log-likelihood estimator')
    parser.add_argument('-p', '--path', help='model path')
    parser.add_argument('-s', '--sigma', default = None)
    parser.add_argument('-d', '--dataset', choices=['mnist', 'tfd'])
    parser.add_argument('-f', '--fold', default = 0, type=int)
    parser.add_argument('-v', '--valid', default = False, action='store_true')
    parser.add_argument('-n', '--num_samples', default=10000, type=int)
    parser.add_argument('-l', '--limit_size', default=1000, type=int)
    parser.add_argument('-b', '--batch_size', default=100, type=int)
    parser.add_argument('-c', '--cross_val', default=10, type=int,
                            help="Number of cross valiation folds")
    parser.add_argument('--sigma_start', default=-1, type=float)
    parser.add_argument('--sigma_end', default=0., type=float)
    args = parser.parse_args()

    # load model
    model = serial.load(args.path)
    src = model.dataset_yaml_src
    batch_size = args.batch_size
    model.set_batch_size(batch_size)

    # load test set
    test = yaml_parse.load(src)
    test = get_test(args.dataset, test, args.fold)

    # generate samples
    samples = model.generator.sample(args.num_samples).eval()
    output_space = model.generator.mlp.get_output_space()
    if 'Conv2D' in str(output_space):
        samples = output_space.convert(samples, output_space.axes, ('b', 0, 1, 'c'))
        samples = samples.reshape((samples.shape[0], numpy.prod(samples.shape[1:])))
    del model
    gc.collect()

    # cross validate sigma
    if args.sigma is None:
        valid = get_valid(args.dataset, limit_size = args.limit_size, fold = args.fold)
        sigma_range = numpy.logspace(args.sigma_start, args.sigma_end, num=args.cross_val)
        sigma = cross_validate_sigma(samples, valid, sigma_range, batch_size)
    else:
        sigma = float(args.sigma)

    print "Using Sigma: {}".format(sigma)
    gc.collect()

    # fit and evaulate
    parzen = theano_parzen(samples, sigma)
    ll = get_nll(test.X, parzen, batch_size = batch_size)
    se = ll.std() / numpy.sqrt(test.X.shape[0])

    print "Log-Likelihood of test set = {}, se: {}".format(ll.mean(), se)

    # valid
    if args.valid:
        valid = get_valid(args.dataset)
        ll = get_nll(valid, parzen, batch_size = batch_size)
        se = ll.std() / numpy.sqrt(val.shape[0])
        print "Log-Likelihood of valid set = {}, se: {}".format(ll.mean(), se)


if __name__ == "__main__":
    main()
