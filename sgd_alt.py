"""
Copy of pylearn2's sgd.py, hacked to support alternating between
epochs of updating only the discriminator and epochs of updating
both discriminator and generator. Ideally this would
be accomplished using pylearn2's FixedVarDescr implementation,
but it is currently not very well supported.
"""
from __future__ import division

__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow, David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "David Warde-Farley"
__email__ = "pylearn-dev@googlegroups"

import logging
import warnings
import numpy as np

from theano import config
from theano import function
from theano.compat.python2x import OrderedDict
from theano.gof.op import get_debug_values

from pylearn2.monitor import Monitor
from pylearn2.space import CompositeSpace, NullSpace
from pylearn2.train_extensions import TrainExtension
from pylearn2.training_algorithms.training_algorithm import TrainingAlgorithm
from pylearn2.training_algorithms.learning_rule import Momentum
from pylearn2.training_algorithms.learning_rule import MomentumAdjustor \
        as LRMomentumAdjustor
from pylearn2.utils.iteration import is_stochastic, has_uniform_batch_size
from pylearn2.utils import py_integer_types, py_float_types
from pylearn2.utils import safe_zip
from pylearn2.utils import serial
from pylearn2.utils import sharedX
from pylearn2.utils.data_specs import DataSpecsMapping
from pylearn2.utils.timing import log_timing
from pylearn2.utils.rng import make_np_rng


log = logging.getLogger(__name__)


class SGD(TrainingAlgorithm):
    """
    SGD = (Minibatch) Stochastic Gradient Descent.
    A TrainingAlgorithm that does stochastic gradient descent on minibatches
    of training examples.

    For theoretical background on this algorithm, see Yoshua Bengio's machine
    learning course notes on the subject:

    http://www.iro.umontreal.ca/~pift6266/H10/notes/gradient.html

    Parameters
    ----------
    learning_rate : float
        The learning rate to use. Train object callbacks can change the
        learning rate after each epoch. SGD update_callbacks can change
        it after each minibatch.
    cost : pylearn2.costs.cost.Cost, optional
        Cost object specifying the objective function to be minimized.
        Optionally, may be None. In this case, SGD will call the model's
        get_default_cost method to obtain the objective function.
    batch_size : int, optional
        The size of the batch to be used.
        If not specified, the model will be asked for the batch size, so
        you must have specified the batch size there.
        (Some models are rigidly defined to only work with one batch size)
    monitoring_batch_size : int, optional
        The size of the monitoring batches.
    monitoring_batches : int, optional
        At the start of each epoch, we run "monitoring", to evaluate
        quantities such as the validation set error.
        monitoring_batches, if specified, determines the number of batches
        to draw from the iterator for each monitoring dataset.
        Unnecessary if not using monitoring or if `monitor_iteration_mode`
        is 'sequential' and `batch_size` is specified (number of
        batches will be calculated based on full dataset size).
        TODO: make it possible to specify different monitoring_batches
        for each monitoring dataset. The Monitor itself already supports
        this.
    monitoring_dataset : Dataset or dictionary, optional
        If not specified, no monitoring is used.
        If specified to be a Dataset, monitor on that Dataset.
        If specified to be dictionary, the keys should be string names
        of datasets, and the values should be Datasets. All monitoring
        channels will be computed for all monitoring Datasets and will
        have the dataset name and an underscore prepended to them.
    monitor_iteration_mode : str, optional
        The iteration mode used to iterate over the examples in all
        monitoring datasets. If not specified, defaults to 'sequential'.
        TODO: make it possible to specify different modes for different
        datasets.
    termination_criterion : instance of \
        pylearn2.termination_criteria.TerminationCriterion, optional

        Used to determine when the algorithm should stop running.
        If not specified, runs forever--or more realistically, until
        external factors halt the python process (Kansas 1977).
    update_callbacks : list, optional
        If specified, each member of the list should be a callable that
        accepts an SGD instance as its only argument.
        All callbacks will be called with this SGD instance after each
        SGD step.
    learning_rule : training_algorithms.learning_rule.LearningRule, optional
        A learning rule computes the new parameter values given old
        parameters and first-order gradients. If learning_rule is None,
        sgd.SGD will update parameters according to the standard SGD
        learning rule:

        .. code-block:: none

            param := param - learning_rate * d cost / d param

        This argument allows more sophisticated learning rules, such
        as SGD with momentum.
    init_momentum : float, **DEPRECATED** option
        Use learning_rule instead.
        If None, does not use momentum otherwise, use momentum and
        initialize the momentum coefficient to init_momentum. Callbacks
        can change this over time just like the learning rate. If the
        gradient is the same on every step, then the update taken by the
        SGD algorithm is scaled by a factor of 1/(1-momentum). See
        section 9 of Geoffrey Hinton's "A Practical Guide to Training
        Restricted Boltzmann Machines" for details.
    set_batch_size : bool, optional
        Defaults to False.
        If True, and batch_size conflicts with model.force_batch_size,
        will call model.set_batch_size(batch_size) in an attempt to
        change model.force_batch_size
    train_iteration_mode : str, optional
        Defaults to 'shuffled_sequential'.
        The iteration mode to use for iterating through training examples.
    batches_per_iter : int, optional
        The number of batches to draw from the iterator over training
        examples.
        If iteration mode is 'sequential' or 'shuffled_sequential', this
        is unnecessary; when unspecified we will iterate over all examples.
    theano_function_mode : a valid argument to theano.function's \
        'mode' parameter, optional

        The theano mode to compile the updates function with. Note that
        pylearn2 includes some wraplinker modes that are not bundled with
        theano. See pylearn2.devtools. These extra modes let you do
        things like check for NaNs at every step, or record md5 digests
        of all computations performed by the update function to help
        isolate problems with nondeterminism.
    monitoring_costs : list, optional
        a list of Cost instances. The Monitor will also include all
        channels defined by these Costs, even though we don't train
        using them.
    seed : valid argument to np.random.RandomState, optional
        The seed used for the random number generate to be passed to the
        training dataset iterator (if any)
    """
    def __init__(self, learning_rate, cost=None, batch_size=None,
                 monitoring_batch_size=None, monitoring_batches=None,
                 monitoring_dataset=None, monitor_iteration_mode='sequential',
                 termination_criterion=None, update_callbacks=None,
                 learning_rule = None, init_momentum = None,
                 set_batch_size = False,
                 train_iteration_mode = None, batches_per_iter=None,
                 theano_function_mode = None, monitoring_costs=None,
                 seed=[2012, 10, 5], discriminator_steps=1):

        self.discriminator_steps = discriminator_steps
        self.train_generator = 0

        if isinstance(cost, (list, tuple, set)):
            raise TypeError("SGD no longer supports using collections of " +
                            "Costs to represent a sum of Costs. Use " +
                            "pylearn2.costs.cost.SumOfCosts instead.")

        if init_momentum:
            warnings.warn("init_momentum interface is deprecated and will "
            "become officially unsuported as of May 9, 2014. Please use the "
            "`learning_rule` parameter instead, providing an object of type "
            "`pylearn2.training_algorithms.learning_rule.Momentum` instead")
            # Convert to new interface under the hood.
            self.learning_rule = Momentum(init_momentum)
        else:
            self.learning_rule = learning_rule

        self.learning_rate = sharedX(learning_rate, 'learning_rate')
        self.cost = cost
        self.batch_size = batch_size
        self.set_batch_size = set_batch_size
        self.batches_per_iter = batches_per_iter
        self._set_monitoring_dataset(monitoring_dataset)
        self.monitoring_batch_size = monitoring_batch_size
        self.monitoring_batches = monitoring_batches
        self.monitor_iteration_mode = monitor_iteration_mode
        if monitoring_dataset is None:
            if monitoring_batch_size is not None:
                raise ValueError("Specified a monitoring batch size " +
                                 "but not a monitoring dataset.")
            if monitoring_batches is not None:
                raise ValueError("Specified an amount of monitoring batches " +
                                 "but not a monitoring dataset.")
        self.termination_criterion = termination_criterion
        self._register_update_callbacks(update_callbacks)
        if train_iteration_mode is None:
            train_iteration_mode = 'shuffled_sequential'
        self.train_iteration_mode = train_iteration_mode
        self.first = True
        self.rng = make_np_rng(seed, which_method=["randn","randint"])
        self.theano_function_mode = theano_function_mode
        self.monitoring_costs = monitoring_costs

    def setup(self, model, dataset):
        """
        Compiles the theano functions needed for the train method.

        Parameters
        ----------
        model : a Model instance
        dataset : Dataset
        """
        if self.cost is None:
            self.cost = model.get_default_cost()

        inf_params = [param for param in model.get_params()
                      if np.any(np.isinf(param.get_value()))]
        if len(inf_params) > 0:
            raise ValueError("These params are Inf: "+str(inf_params))
        if any([np.any(np.isnan(param.get_value()))
                for param in model.get_params()]):
            nan_params = [param for param in model.get_params()
                          if np.any(np.isnan(param.get_value()))]
            raise ValueError("These params are NaN: "+str(nan_params))
        self.model = model

        self._synchronize_batch_size(model)
        model._test_batch_size = self.batch_size
        self.monitor = Monitor.get_monitor(model)
        self.monitor._sanity_check()

        # test if force batch size and batch size
        if getattr(model, "force_batch_size", False) and \
           any(dataset.get_design_matrix().shape[0] % self.batch_size != 0 for
               dataset in self.monitoring_dataset.values()) and \
           not has_uniform_batch_size(self.monitor_iteration_mode):

            raise ValueError("Dataset size is not a multiple of batch size."
                             "You should set monitor_iteration_mode to "
                             "even_sequential, even_shuffled_sequential or "
                             "even_batchwise_shuffled_sequential")

        data_specs = self.cost.get_data_specs(self.model)
        mapping = DataSpecsMapping(data_specs)
        space_tuple = mapping.flatten(data_specs[0], return_tuple=True)
        source_tuple = mapping.flatten(data_specs[1], return_tuple=True)

        # Build a flat tuple of Theano Variables, one for each space.
        # We want that so that if the same space/source is specified
        # more than once in data_specs, only one Theano Variable
        # is generated for it, and the corresponding value is passed
        # only once to the compiled Theano function.
        theano_args = []
        for space, source in safe_zip(space_tuple, source_tuple):
            name = '%s[%s]' % (self.__class__.__name__, source)
            arg = space.make_theano_batch(name=name,
                                          batch_size=self.batch_size)
            theano_args.append(arg)
        theano_args = tuple(theano_args)

        # Methods of `self.cost` need args to be passed in a format compatible
        # with data_specs
        nested_args = mapping.nest(theano_args)
        fixed_var_descr = self.cost.get_fixed_var_descr(model, nested_args)
        self.on_load_batch = fixed_var_descr.on_load_batch

        cost_value = self.cost.expr(model, nested_args,
                                    ** fixed_var_descr.fixed_vars)

        if cost_value is not None and cost_value.name is None:
            # Concatenate the name of all tensors in theano_args !?
            cost_value.name = 'objective'

        # Set up monitor to model the objective value, learning rate,
        # momentum (if applicable), and extra channels defined by
        # the cost
        learning_rate = self.learning_rate
        if self.monitoring_dataset is not None:
            if (self.monitoring_batch_size is None and
                    self.monitoring_batches is None):
                self.monitoring_batch_size = self.batch_size
                self.monitoring_batches = self.batches_per_iter
            self.monitor.setup(dataset=self.monitoring_dataset,
                               cost=self.cost,
                               batch_size=self.monitoring_batch_size,
                               num_batches=self.monitoring_batches,
                               extra_costs=self.monitoring_costs,
                               mode=self.monitor_iteration_mode)
            dataset_name = self.monitoring_dataset.keys()[0]
            monitoring_dataset = self.monitoring_dataset[dataset_name]
            #TODO: have Monitor support non-data-dependent channels
            self.monitor.add_channel(name='learning_rate',
                                     ipt=None,
                                     val=learning_rate,
                                     data_specs=(NullSpace(), ''),
                                     dataset=monitoring_dataset)

            if self.learning_rule:
                self.learning_rule.add_channels_to_monitor(
                        self.monitor,
                        monitoring_dataset)

        params = list(model.get_params())
        assert len(params) > 0
        for i, param in enumerate(params):
            if param.name is None:
                param.name = 'sgd_params[%d]' % i
        self.params = params


        grads, updates = self.cost.get_gradients(model, nested_args,
                                                 ** fixed_var_descr.fixed_vars)
        if not isinstance(grads, OrderedDict):
            raise TypeError(str(type(self.cost)) + ".get_gradients returned " +
                            "something with" + str(type(grads)) + "as its " +
                            "first member. Expected OrderedDict.")

        for param in grads:
            assert param in params
        for param in params:
            assert param in grads

        lr_scalers = model.get_lr_scalers()

        for key in lr_scalers:
            if key not in params:
                raise ValueError("Tried to scale the learning rate on " +\
                        str(key)+" which is not an optimization parameter.")

        assert len(updates.keys()) == 0

        def get_func(learn_discriminator, learn_generator, dont_you_fucking_dare_touch_the_generator=False):

            updates = OrderedDict()

            assert (learn_discriminator or learn_generator) and not (learn_discriminator and learn_generator)

            if learn_discriminator:
                cur_params = model.discriminator.get_params()
            else:
                cur_params = model.generator.get_params()

            def check():
                for param in params:
                    if param not in cur_params:
                        assert param not in updates

            cur_grads = OrderedDict()
            for param in cur_params:
                cur_grads[param] = grads[param]

            for param in grads:
                if grads[param].name is None and cost_value is not None:
                    grads[param].name = ('grad(%(costname)s, %(paramname)s)' %
                                         {'costname': cost_value.name,
                                          'paramname': param.name})
                assert grads[param].dtype == param.dtype

            cur_lr_scalers = OrderedDict()
            for param in cur_params:
                if param in lr_scalers:
                    lr_scaler = lr_scalers[param]
                    cur_lr_scalers[param] = lr_scaler

            log.info('Parameter and initial learning rate summary:')
            for param in cur_params:
                param_name = param.name
                if param_name is None:
                    param_name = 'anon_param'
                lr = learning_rate.get_value() * cur_lr_scalers.get(param,1.)
                log.info('\t' + param_name + ': ' + str(lr))

            updates.update(self.learning_rule.get_updates(
                    learning_rate, cur_grads, cur_lr_scalers))
            check()

            for param in cur_params:
                if updates[param].name is None:
                    updates[param].name = 'sgd_update(' + param.name + ')'
            check()
            model.modify_updates(updates)
            check()
            for param in cur_params:
                update = updates[param]
                if update.name is None:
                    update.name = 'censor(sgd_update(' + param.name + '))'
                for update_val in get_debug_values(update):
                    if np.any(np.isinf(update_val)):
                        raise ValueError("debug value of %s contains infs" %
                                update.name)
                    if np.any(np.isnan(update_val)):
                        raise ValueError("debug value of %s contains nans" %
                                update.name)

            check()

            if dont_you_fucking_dare_touch_the_generator:
                for param in model.generator.get_params():
                    assert param not in updates

            with log_timing(log, 'Compiling sgd_update'):
                return function(theano_args,
                                           updates=updates,
                                           name='sgd_update',
                                           on_unused_input='ignore',
                                           mode=self.theano_function_mode)
        self.d_func = get_func(1, 0, dont_you_fucking_dare_touch_the_generator=True)
        self.g_func = get_func(0, 1)

    def train(self, dataset):
        """
        Runs one epoch of SGD training on the specified dataset.

        Parameters
        ----------
        dataset : Dataset
        """


        if not hasattr(self, 'd_func'):
            raise Exception("train called without first calling setup")

        # Make sure none of the parameters have bad values
        for param in self.params:
            value = param.get_value(borrow=True)
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                raise Exception("NaN in " + param.name)

        self.first = False
        rng = self.rng
        if not is_stochastic(self.train_iteration_mode):
            rng = None

        data_specs = self.cost.get_data_specs(self.model)

        # The iterator should be built from flat data specs, so it returns
        # flat, non-redundent tuples of data.
        mapping = DataSpecsMapping(data_specs)
        space_tuple = mapping.flatten(data_specs[0], return_tuple=True)
        source_tuple = mapping.flatten(data_specs[1], return_tuple=True)
        if len(space_tuple) == 0:
            # No data will be returned by the iterator, and it is impossible
            # to know the size of the actual batch.
            # It is not decided yet what the right thing to do should be.
            raise NotImplementedError("Unable to train with SGD, because "
                    "the cost does not actually use data from the data set. "
                    "data_specs: %s" % str(data_specs))
        flat_data_specs = (CompositeSpace(space_tuple), source_tuple)

        iterator = dataset.iterator(mode=self.train_iteration_mode,
                batch_size=self.batch_size,
                data_specs=flat_data_specs, return_tuple=True,
                rng = rng, num_batches = self.batches_per_iter)


        on_load_batch = self.on_load_batch
        i = 0
        for batch in iterator:
            for callback in on_load_batch:
                callback(*batch)
            if self.train_generator and i == self.discriminator_steps:
                self.g_func(*batch)
                i = 0
            else:
                self.d_func(*batch)
                i += 1
            # iterator might return a smaller batch if dataset size
            # isn't divisible by batch_size
            # Note: if data_specs[0] is a NullSpace, there is no way to know
            # how many examples would actually have been in the batch,
            # since it was empty, so actual_batch_size would be reported as 0.
            actual_batch_size = flat_data_specs[0].np_batch_size(batch)
            self.monitor.report_batch(actual_batch_size)
            for callback in self.update_callbacks:
                callback(self)


        # Make sure none of the parameters have bad values
        for param in self.params:
            value = param.get_value(borrow=True)
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                raise Exception("NaN in " + param.name)

        self.train_generator = not self.train_generator

    def continue_learning(self, model):
        """
        Returns True if the algorithm should continue running, or False
        if it has reached convergence / started overfitting and should
        stop.

        Parameters
        ----------
        model : a Model instance
        """
        if self.termination_criterion is None:
            return True
        else:
            return self.termination_criterion.continue_learning(self.model)

class MonitorBasedLRAdjuster(TrainExtension):
    """
    A TrainExtension that uses the on_monitor callback to adjust
    the learning rate on each epoch. It pulls out a channel
    from the model's monitor and adjusts the learning rate
    based on what happened to the monitoring channel on the last
    epoch. If the channel is greater than high_trigger times
    its previous value, the learning rate will be scaled by
    shrink_amt (which should be < 1 for this scheme to make
    sense). The idea is that in this case the learning algorithm
    is overshooting the bottom of the objective function.

    If the objective is less than high_trigger but
    greater than low_trigger times its previous value, the
    learning rate will be scaled by grow_amt (which should be > 1
    for this scheme to make sense). The idea is that the learning
    algorithm is making progress but at too slow of a rate.

    Parameters
    ----------
    high_trigger : float, optional
        See class-level docstring
    low_trigger : float, optional
        See class-level docstring
    grow_amt : float, optional
        See class-level docstring
    min_lr : float, optional
        All updates to the learning rate are clipped to be at least
        this value.
    max_lr : float, optional
        All updates to the learning rate are clipped to be at most
        this value.
    dataset_name : str, optional
        If specified, use dataset_name + "_objective" as the channel
        to guide the learning rate adaptation.
    channel_name : str, optional
        If specified, use channel_name as the channel to guide the
        learning rate adaptation. Conflicts with dataset_name.
        If neither dataset_name nor channel_name is specified, uses
        "objective"
    """

    def __init__(self, high_trigger=1., shrink_amt=.99,
                 low_trigger=.99, grow_amt=1.01,
                 min_lr = 1e-7, max_lr = 1.,
                 dataset_name=None, channel_name=None):
        self.high_trigger = high_trigger
        self.shrink_amt = shrink_amt
        self.low_trigger = low_trigger
        self.grow_amt = grow_amt
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.dataset_name = None
        if channel_name is not None:
            self.channel_name = channel_name
        else:
            if dataset_name is not None:
                self.channel_name = dataset_name + '_objective'
                self.dataset_name = dataset_name
            else:
                self.channel_name = None

    def on_monitor(self, model, dataset, algorithm):
        """
        Adjusts the learning rate based on the contents of model.monitor

        Parameters
        ----------
        model : a Model instance
        dataset : Dataset
        algorithm : WRITEME
        """
        model = algorithm.model
        lr = algorithm.learning_rate
        current_learning_rate = lr.get_value()
        assert hasattr(model, 'monitor'), ("no monitor associated with "
                + str(model))
        monitor = model.monitor
        monitor_channel_specified = True

        if self.channel_name is None:
            monitor_channel_specified = False
            channels = [elem for elem in monitor.channels
                    if elem.endswith("objective")]
            if len(channels) < 1:
                raise ValueError("There are no monitoring channels that end "
                        "with \"objective\". Please specify either "
                        "channel_name or dataset_name.")
            elif len(channels) > 1:
                datasets = algorithm.monitoring_dataset.keys()
                raise ValueError("There are multiple monitoring channels that"
                        "end with \"_objective\". The list of available "
                        "datasets are: " +
                                str(datasets) + " . Please specify either "
                                "channel_name or dataset_name in the "
                                "MonitorBasedLRAdjuster constructor to "
                                'disambiguate.')
            else:
                self.channel_name = channels[0]
                warnings.warn('The channel that has been chosen for '
                        'monitoring is: ' +
                              str(self.channel_name) + '.')

        try:
            v = monitor.channels[self.channel_name].val_record
        except KeyError:
            err_input = ''
            if monitor_channel_specified:
                if self.dataset_name:
                    err_input = 'The dataset_name \'' + str(
                            self.dataset_name) + '\' is not valid.'
                else:
                    err_input = 'The channel_name \'' + str(
                            self.channel_name) + '\' is not valid.'
            err_message = 'There is no monitoring channel named \'' + \
                    str(self.channel_name) + '\'. You probably need to ' + \
                    'specify a valid monitoring channel by using either ' + \
                    'dataset_name or channel_name in the ' + \
                    'MonitorBasedLRAdjuster constructor. ' + err_input
            raise ValueError(err_message)

        if len(v) < 1:
            if monitor.dataset is None:
                assert len(v) == 0
                raise ValueError("You're trying to use a monitor-based "
                        "learning rate adjustor but the monitor has no "
                        "entries because you didn't specify a "
                        "monitoring dataset.")

            raise ValueError("For some reason there are no monitor entries"
                                 "yet the MonitorBasedLRAdjuster has been "
                                 "called. This should never happen. The Train"
                                 " object should call the monitor once on "
                                 "initialization, then call the callbacks. "
                                 "It seems you are either calling the "
                                 "callback manually rather than as part of a "
                                 "training algorithm, or there is a problem "
                                "with the Train object.")
        if len(v) == 1:
            #only the initial monitoring has happened
            #no learning has happened, so we can't adjust the learning rate yet
            #just do nothing
            return

        rval = current_learning_rate

        log.info("monitoring channel is {0}".format(self.channel_name))

        if v[-1] > self.high_trigger * v[-2]:
            rval *= self.shrink_amt
            log.info("shrinking learning rate to %f" % rval)
        elif v[-1] > self.low_trigger * v[-2]:
            rval *= self.grow_amt
            log.info("growing learning rate to %f" % rval)

        rval = max(self.min_lr, rval)
        rval = min(self.max_lr, rval)

        lr.set_value(np.cast[lr.dtype](rval))


class PatienceBasedTermCrit(object):
    """
    A monitor-based termination criterion using a geometrically increasing
    amount of patience. If the selected channel has decreased by a certain
    proportion when comparing to the lowest value seen yet, the patience is
    set to a factor of the number of examples seen, which by default
    (patience_increase=2.) ensures the model has seen as many examples as the
    number of examples that lead to the lowest value before concluding a local
    optima has been reached.

    Note: Technically, the patience corresponds to a number of epochs to be
    independent of the size of the dataset, so be aware of that when choosing
    initial_patience.

    Parameters
    ----------
    prop_decrease : float
        The factor X in the (1 - X) * best_value threshold
    initial_patience : int
        Minimal number of epochs the model has to run before it can stop
    patience_increase : float, optional
        The factor X in the patience = X * n_iter update.
    channel_name : string, optional
        Name of the channel to examine. If None and the monitor
        has only one channel, this channel will be used; otherwise, an
        error will be raised.
    """
    def __init__(self, prop_decrease, initial_patience,
                 patience_increase=2., channel_name=None):
        self._channel_name = channel_name
        self.prop_decrease = prop_decrease
        self.patience = initial_patience
        self.best_value = np.inf
        self.patience_increase = patience_increase

    def __call__(self, model):
        """
        Returns True or False depending on whether the optimization should
        stop or not. The optimization should stop if it has run for a number
        of epochs superior to the patience without any improvement.

        Parameters
        ----------
        model : Model
            The model used in the experiment and from which the monitor used
            in the termination criterion will be extracted.

        Returns
        -------
        bool
            True or False, indicating if the optimization should stop or not.
        """
        monitor = model.monitor
        # In the case the monitor has only one channel, the channel_name can
        # be omitted and the criterion will examine the only channel
        # available. However, if the monitor has multiple channels, leaving
        # the channel_name unspecified will raise an error.
        if self._channel_name is None:
            if len(monitor.channels) != 1:
                raise ValueError("Only single-channel monitors are supported "
                                 "for channel_name == None")
            v = monitor.channels.values()[0].val_record
        else:
            v = monitor.channels[self._channel_name].val_record
        # If the channel value decrease is higher than the threshold, we
        # update the best value to this value and we update the patience.
        if v[-1] < self.best_value * (1. - self.prop_decrease):
            # Using the max between actual patience and updated patience
            # ensures that the model will run for at least the initial
            # patience and that it would behave correctly if the user
            # chooses a dumb value (i.e. less than 1)
            self.patience = max(self.patience, len(v) * self.patience_increase)
            self.best_value = v[-1]

        return len(v) < self.patience


class AnnealedLearningRate(object):
    """
    This is a callback for the SGD algorithm rather than the Train object.
    This anneals the learning rate to decrease as 1/t where t is the number
    of gradient descent updates done so far. Use OneOverEpoch as Train object
    callback if you would prefer 1/t where t is epochs.

    Parameters
    ----------
    anneal_start : int
        The epoch on which to begin annealing
    """
    def __init__(self, anneal_start):
        self._initialized = False
        self._count = 0
        self._anneal_start = anneal_start

    def __call__(self, algorithm):
        """
        Updates the learning rate according to the annealing schedule.

        Parameters
        ----------
        algorithm : WRITEME
        """
        if not self._initialized:
            self._base = algorithm.learning_rate.get_value()
        self._count += 1
        algorithm.learning_rate.set_value(self.current_learning_rate())

    def current_learning_rate(self):
        """
        Returns the current desired learning rate according to the
        annealing schedule.
        """
        return self._base * min(1, self._anneal_start / self._count)

class ExponentialDecay(object):
    """
    This is a callback for the `SGD` algorithm rather than the `Train` object.
    This anneals the learning rate by dividing by decay_factor after each
    gradient descent step. It will not shrink the learning rate beyond
    `min_lr`.

    Parameters
    ----------
    decay_factor : float
        The learning rate at step t is given by
        `init_learning_rate / (decay_factor ** t)`
    min_lr : float
        The learning rate will be clipped to be at least this value
    """

    def __init__(self, decay_factor, min_lr):
        if isinstance(decay_factor, str):
            decay_factor = float(decay_factor)
        if isinstance(min_lr, str):
            min_lr = float(min_lr)
        assert isinstance(decay_factor, float)
        assert isinstance(min_lr, float)
        self.__dict__.update(locals())
        del self.self
        self._count = 0
        self._min_reached = False

    def __call__(self, algorithm):
        """
        Updates the learning rate according to the exponential decay schedule.

        Parameters
        ----------
        algorithm : SGD
            The SGD instance whose `learning_rate` field should be modified.
        """
        if self._count == 0:
            self._base_lr = algorithm.learning_rate.get_value()
        self._count += 1

        if not self._min_reached:
            # If we keep on executing the exponentiation on each mini-batch,
            # we will eventually get an OverflowError. So make sure we
            # only do the computation until min_lr is reached.
            new_lr = self._base_lr / (self.decay_factor ** self._count)
            if new_lr <= self.min_lr:
                self._min_reached = True
                new_lr = self.min_lr
        else:
            new_lr = self.min_lr

        new_lr = np.cast[config.floatX](new_lr)
        algorithm.learning_rate.set_value(new_lr)

class LinearDecay(object):
    """
    This is a callback for the SGD algorithm rather than the Train object.
    This anneals the learning rate to decay_factor times of the initial value
    during time start till saturate.

    Parameters
    ----------
    start : int
        The step at which to start decreasing the learning rate
    saturate : int
        The step at which to stop decreating the learning rate
    decay_factor : float
        `final learning rate = decay_factor * initial learning rate`
    """

    def __init__(self, start, saturate, decay_factor):
        if isinstance(decay_factor, str):
            decay_factor = float(decay_factor)
        if isinstance(start, str):
            start = float(start)
        if isinstance(saturate, str):
            saturate = float(saturate)
        assert isinstance(decay_factor, float)
        assert isinstance(start, (py_integer_types, py_float_types))
        assert isinstance(saturate, (py_integer_types, py_float_types))
        assert saturate > start
        assert start > 0
        self.__dict__.update(locals())
        del self.self
        self._count = 0

    def __call__(self, algorithm):
        """
        Adjusts the learning rate according to the linear decay schedule

        Parameters
        ----------
        algorithm : WRITEME
        """
        if self._count == 0:
            self._base_lr = algorithm.learning_rate.get_value()
            self._step = ((self._base_lr - self._base_lr * self.decay_factor) /
                          (self.saturate - self.start + 1))
        self._count += 1
        if self._count >= self.start:
            if self._count < self.saturate:
                new_lr = self._base_lr - self._step * (self._count
                        - self.start + 1)
            else:
                new_lr = self._base_lr * self.decay_factor
        else:
            new_lr = self._base_lr
        assert new_lr > 0
        new_lr = np.cast[config.floatX](new_lr)
        algorithm.learning_rate.set_value(new_lr)


def MomentumAdjustor(final_momentum, start, saturate):
    """
    Deprecated class used with the deprecated init_momentum argument.
    Use learning_rule.MomentumAdjustor instead.

    Parameters
    ----------
    final_momentum : WRITEME
    start : WRITEME
    saturate : WRITEME
    """
    warnings.warn("sgd.MomentumAdjustor interface is deprecated and will "
    "become officially unsupported as of May 9, 2014. Please use "
    "`learning_rule.MomentumAdjustor` instead.")
    return LRMomentumAdjustor(final_momentum, start, saturate)


class OneOverEpoch(TrainExtension):
    """
    Scales the learning rate like one over # epochs

    Parameters
    ----------
    start : int
        The epoch on which to start shrinking the learning rate
    half_life : int, optional
        How many epochs after start it will take for the learning rate to lose
        half its value for the first time (to lose the next half of its value
        will take twice as long)
    min_lr : float, optional
        The minimum value the learning rate can take on
    """
    def __init__(self, start, half_life = None, min_lr = 1e-6):
        self.__dict__.update(locals())
        del self.self
        self._initialized = False
        self._count = 0
        assert start >= 0
        if half_life is None:
            self.half_life = start + 1
        else:
            assert half_life > 0

    def on_monitor(self, model, dataset, algorithm):
        """
        Adjusts the learning rate according to the decay schedule.

        Parameters
        ----------
        model : a Model instance
        dataset : Dataset
        algorithm : WRITEME
        """

        if not self._initialized:
            self._init_lr = algorithm.learning_rate.get_value()
            if self._init_lr < self.min_lr:
                raise ValueError("The initial learning rate is smaller than " +
                                 "the minimum allowed learning rate.")
            self._initialized = True
        self._count += 1
        algorithm.learning_rate.set_value(np.cast[config.floatX](
            self.current_lr()))

    def current_lr(self):
        """
        Returns the learning rate currently desired by the decay schedule.
        """
        if self._count < self.start:
            scale = 1
        else:
            scale = float(self.half_life) / float(self._count - self.start
                    + self.half_life)
        lr = self._init_lr * scale
        clipped = max(self.min_lr, lr)
        return clipped

class LinearDecayOverEpoch(TrainExtension):
    """
    Scales the learning rate linearly on each epochs

    Parameters
    ----------
    start : int
        The epoch on which to start shrinking the learning rate
    saturate : int
        The epoch to saturate the shrinkage
    decay_factor : float
        The final value would be initial learning rate times decay_factor
    """

    def __init__(self, start, saturate, decay_factor):
        self.__dict__.update(locals())
        del self.self
        self._initialized = False
        self._count = 0
        assert isinstance(decay_factor, float)
        assert isinstance(start, (py_integer_types, py_float_types))
        assert isinstance(saturate, (py_integer_types, py_float_types))
        assert saturate > start
        assert start >= 0
        assert saturate >= start

    def on_monitor(self, model, dataset, algorithm):
        """
        Updates the learning rate based on the linear decay schedule.

        Parameters
        ----------
        model : a Model instance
        dataset : Dataset
        algorithm : WRITEME
        """
        if not self._initialized:
            self._init_lr = algorithm.learning_rate.get_value()
            self._step = ((self._init_lr - self._init_lr * self.decay_factor) /
                          (self.saturate - self.start + 1))
            self._initialized = True
        self._count += 1
        algorithm.learning_rate.set_value(np.cast[config.floatX](
            self.current_lr()))

    def current_lr(self):
        """
        Returns the learning rate currently desired by the decay schedule.
        """
        if self._count >= self.start:
            if self._count < self.saturate:
                new_lr = self._init_lr - self._step * (self._count
                        - self.start + 1)
            else:
                new_lr = self._init_lr * self.decay_factor
        else:
            new_lr = self._init_lr
        assert new_lr > 0
        return new_lr

class _PolyakWorker(object):
    """
    Only to be used by the PolyakAveraging TrainingCallback below.
    Do not use directly.
    A callback for the SGD class.

    Parameters
    ----------
    model : a Model
        The model whose parameters we want to train with Polyak averaging
    """

    def __init__(self, model):
        avg_updates = OrderedDict()
        t = sharedX(1.)
        self.param_to_mean = OrderedDict()
        for param in model.get_params():
            mean = sharedX(param.get_value())
            assert type(mean) == type(param)
            self.param_to_mean[param] = mean
            avg_updates[mean] = mean - (mean - param) / t
            avg_updates[t] = t + 1.
        self.avg = function([], updates = avg_updates)

    def __call__(self, algorithm):
        """
        To be called after each SGD step.
        Updates the Polyak averaged-parameters for this model

        Parameters
        ----------
        algorithm : WRITEME
        """
        self.avg()

class PolyakAveraging(TrainExtension):
    """
    See "A Tutorial on Stochastic Approximation Algorithms
        for Training Restricted Boltzmann Machines and
        Deep Belief Nets" by Kevin Swersky et al

    This functionality is still a work in progress. Currently,
    your model needs to implement "add_polyak_channels" to
    use it.

    The problem is that Polyak averaging shouldn't modify
    the model parameters. It should keep a second copy
    that it averages in the background. This second copy
    doesn't get to come back in and affect the learning process
    though.

    (IG tried having the second copy get pushed back into
    the model once per epoch, but this turned out to be
    harmful, at least in limited tests)

    So we need a cleaner interface for monitoring the
    averaged copy of the parameters, and we need to make
    sure the saved model at the end uses the averaged
    parameters, not the parameters used for computing
    the gradients during training.

    TODO: make use of the new on_save callback instead
        of duplicating Train's save_freq flag

    Parameters
    ----------
    start : int
        The epoch after which to start averaging (0 = start averaging
        immediately)
    save_path : str, optional
        WRITEME
    save_freq : int, optional
        WRITEME

    Notes
    -----
    This is usually used with a fixed, rather than annealed learning
    rate. It may be used in conjunction with momentum.
    """

    def __init__(self, start, save_path=None, save_freq=1):
        self.__dict__.update(locals())
        del self.self
        self._count = 0
        assert isinstance(start, py_integer_types)
        assert start >= 0

    def on_monitor(self, model, dataset, algorithm):
        """
        Make sure Polyak-averaged model gets monitored.
        Save the model if necessary.

        Parameters
        ----------
        model : a Model instance
        dataset : Dataset
        algorithm : WRITEME
        """
        if self._count == self.start:
            self._worker = _PolyakWorker(model)
            algorithm.update_callbacks.append(self._worker)
            #HACK
            try:
                model.add_polyak_channels(self._worker.param_to_mean,
                                          algorithm.monitoring_dataset)
            except AttributeError:
                pass
        elif self.save_path is not None and self._count > self.start and \
                self._count % self.save_freq == 0:
            saved_params = OrderedDict()
            for param in model.get_params():
                saved_params[param] = param.get_value()
                param.set_value(self._worker.param_to_mean[param].get_value())
            serial.save(self.save_path, model)
            for param in model.get_params():
                param.set_value(saved_params[param])
        self._count += 1
