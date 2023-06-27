import warnings
from typing import Dict, Optional, Any
from .constraints import *
from .extra_utils import sum_rightmost

import megengine.functional as F
import megengine as mge
import numpy as np
import math

from functools import reduce


class Distribution(object):
    r"""
    Distribution is the abstract base class for probability distributions.
    """

    has_rsample = False
    has_enumerate_support = False
    _validate_args = __debug__

    @staticmethod
    def set_default_validate_args(value):
        """
        Sets whether validation is enabled or disabled.

        The default behavior mimics Python's ``assert`` statement: validation
        is on by default, but is disabled if Python is run in optimized mode
        (via ``python -O``). Validation may be expensive, so you may want to
        disable it once a model is working.

        Args:
            value (bool): Whether to enable validation.
        """
        if value not in [True, False]:
            raise ValueError
        Distribution._validate_args = value

    def __init__(self, batch_shape=(), event_shape=(), validate_args=None):
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        if validate_args is not None:
            self._validate_args = validate_args
        super(Distribution, self).__init__()

    def expand(self, batch_shape, _instance=None):
        """
        Returns a new distribution instance (or populates an existing instance
        provided by a derived class) with batch dimensions expanded to
        `batch_shape`. This method calls :class:`~torch.Tensor.expand` on
        the distribution's parameters. As such, this does not allocate new
        memory for the expanded distribution instance. Additionally,
        this does not repeat any args checking or parameter broadcasting in
        `__init__.py`, when an instance is first created.

        Args:
            batch_shape (torch.Size): the desired expanded size.
            _instance: new instance provided by subclasses that
                need to override `.expand`.

        Returns:
            New distribution instance with batch dimensions expanded to
            `batch_size`.
        """
        raise NotImplementedError

    @property
    def batch_shape(self):
        """
        Returns the shape over which parameters are batched.
        """
        return self._batch_shape

    @property
    def event_shape(self):
        """
        Returns the shape of a single sample (without batching).
        """
        return self._event_shape

    @property
    def arg_constraints(self) -> Dict[str, Constraint]:
        """
        Returns a dictionary from argument names to
        :class:`~torch.distributions.constraints.Constraint` objects that
        should be satisfied by each argument of this distribution. Args that
        are not tensors need not appear in this dict.
        """
        raise NotImplementedError

    @property
    def support(self) -> Optional[Any]:
        """
        Returns a :class:`~torch.distributions.constraints.Constraint` object
        representing this distribution's support.
        """
        raise NotImplementedError

    @property
    def mean(self):
        """
        Returns the mean of the distribution.
        """
        raise NotImplementedError

    @property
    def variance(self):
        """
        Returns the variance of the distribution.
        """
        raise NotImplementedError

    @property
    def stddev(self):
        """
        Returns the standard deviation of the distribution.
        """
        return self.variance.sqrt()

    def sample(self, sample_shape):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched.
        """
        return self.rsample(sample_shape)

    def rsample(self, sample_shape):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.
        """
        raise NotImplementedError

    def sample_n(self, n):
        """
        Generates n samples or n batches of samples if the distribution
        parameters are batched.
        """
        warnings.warn('sample_n will be deprecated. Use .sample((n,)) instead', UserWarning)
        return self.sample((n,))

    def log_prob(self, value):
        """
        Returns the log of the probability density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
        raise NotImplementedError

    def cdf(self, value):
        """
        Returns the cumulative density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
        raise NotImplementedError

    def icdf(self, value):
        """
        Returns the inverse cumulative density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
        raise NotImplementedError

    def enumerate_support(self, expand=True):
        """
        Returns tensor containing all values supported by a discrete
        distribution. The result will enumerate over dimension 0, so the shape
        of the result will be `(cardinality,) + batch_shape + event_shape`
        (where `event_shape = ()` for univariate distributions).

        Note that this enumerates over all batched tensors in lock-step
        `[[0, 0], [1, 1], ...]`. With `expand=False`, enumeration happens
        along dim 0, but with the remaining batch dimensions being
        singleton dimensions, `[[0], [1], ..`.

        To iterate over the full Cartesian product use
        `itertools.product(m.enumerate_support())`.

        Args:
            expand (bool): whether to expand the support over the
                batch dims to match the distribution's `batch_shape`.

        Returns:
            Tensor iterating over dimension 0.
        """
        raise NotImplementedError

    def entropy(self):
        """
        Returns entropy of distribution, batched over batch_shape.

        Returns:
            Tensor of shape batch_shape.
        """
        raise NotImplementedError

    def perplexity(self):
        """
        Returns perplexity of distribution, batched over batch_shape.

        Returns:
            Tensor of shape batch_shape.
        """
        return F.exp(self.entropy())

    def _extended_shape(self, sample_shape):
        """
        Returns the size of the sample returned by the distribution, given
        a `sample_shape`. Note, that the batch and event shapes of a distribution
        instance are fixed at the time of construction. If this is empty, the
        returned shape is upcast to (1,).

        Args:
            sample_shape (torch.Size): the size of the sample to be drawn.
        """
        return sample_shape + self._batch_shape + self._event_shape

    def _validate_sample(self, value):
        """
        Argument validation for distribution methods such as `log_prob`,
        `cdf` and `icdf`. The rightmost dimensions of a value to be
        scored via these methods must agree with the distribution's batch
        and event shapes.

        Args:
            value (Tensor): the tensor whose log probability is to be
                computed by the `log_prob` method.
        Raises
            ValueError: when the rightmost dimensions of `value` do not match the
                distribution's batch and event shapes.
        """
        if not isinstance(value, mge.Tensor):
            raise ValueError('The value argument to log_prob must be a Tensor')

        event_dim_start = len(value.shape) - len(self._event_shape)
        if value.shape[event_dim_start:] != self._event_shape:
            raise ValueError('The right-most size of value must match event_shape: {} vs {}.'.
                             format(value.shape, self._event_shape))

        actual_shape = value.shape
        expected_shape = self._batch_shape + self._event_shape
        for i, j in zip(reversed(actual_shape), reversed(expected_shape)):
            if i != 1 and j != 1 and i != j:
                raise ValueError('Value is not broadcastable with batch_shape+event_shape: {} vs {}.'.
                                 format(actual_shape, expected_shape))
        try:
            support = self.support
        except NotImplementedError:
            warnings.warn(f'{self.__class__} does not define `support` to enable ' +
                          'sample validation. Please initialize the distribution with ' +
                          '`validate_args=False` to turn off validation.')
            return
        assert support is not None
        valid = support.check(value)
        if F.sum(1-valid) != 0:
            raise ValueError(
                "Expected value argument "
                f"({type(value).__name__} of shape {tuple(value.shape)}) "
                f"to be within the support ({repr(support)}) "
                f"of the distribution {repr(self)}, "
                f"but found invalid values:\n{value}"
            )

    def _get_checked_instance(self, cls, _instance=None):
        if _instance is None and type(self).__init__ != cls.__init__:
            raise NotImplementedError("Subclass {} of {} that defines a custom __init__ method "
                                      "must also define a custom .expand() method.".
                                      format(self.__class__.__name__, cls.__name__))
        return self.__new__(type(self)) if _instance is None else _instance

    def __repr__(self):
        param_names = [k for k, _ in self.arg_constraints.items() if k in self.__dict__]
        args_string = ', '.join(['{}: {}'.format(p, self.__dict__[p]
                                if get_numel(self.__dict__[p]) == 1
                                else self.__dict__[p].shape) for p in param_names])
        return self.__class__.__name__ + '(' + args_string + ')'


class Independent(Distribution):
    r"""
    Reinterprets some of the batch dims of a distribution as event dims.

    This is mainly useful for changing the shape of the result of
    :meth:`log_prob`. For example to create a diagonal Normal distribution with
    the same shape as a Multivariate Normal distribution (so they are
    interchangeable), you can::

        >>> loc = torch.zeros(3)
        >>> scale = torch.ones(3)
        >>> mvn = MultivariateNormal(loc, scale_tril=torch.diag(scale))
        >>> [mvn.batch_shape, mvn.event_shape]
        [torch.Size(()), torch.Size((3,))]
        >>> normal = Normal(loc, scale)
        >>> [normal.batch_shape, normal.event_shape]
        [torch.Size((3,)), torch.Size(())]
        >>> diagn = Independent(normal, 1)
        >>> [diagn.batch_shape, diagn.event_shape]
        [torch.Size(()), torch.Size((3,))]

    Args:
        base_distribution (torch.distributions.distribution.Distribution): a
            base distribution
        reinterpreted_batch_ndims (int): the number of batch dims to
            reinterpret as event dims
    """
    arg_constraints: Dict[str, Constraint] = {}

    def __init__(self, base_distribution, reinterpreted_batch_ndims, validate_args=None):
        if reinterpreted_batch_ndims > len(base_distribution.batch_shape):
            raise ValueError("Expected reinterpreted_batch_ndims <= len(base_distribution.batch_shape), "
                             "actual {} vs {}".format(reinterpreted_batch_ndims,
                                                      len(base_distribution.batch_shape)))
        shape = base_distribution.batch_shape + base_distribution.event_shape
        event_dim = reinterpreted_batch_ndims + len(base_distribution.event_shape)
        batch_shape = shape[:len(shape) - event_dim]
        event_shape = shape[len(shape) - event_dim:]
        self.base_dist = base_distribution
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        super(Independent, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Independent, _instance)
        new.base_dist = self.base_dist.expand(batch_shape +
                                              self.event_shape[:self.reinterpreted_batch_ndims])
        new.reinterpreted_batch_ndims = self.reinterpreted_batch_ndims
        super(Independent, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def has_rsample(self):
        return self.base_dist.has_rsample

    @property
    def has_enumerate_support(self):
        if self.reinterpreted_batch_ndims > 0:
            return False
        return self.base_dist.has_enumerate_support

    # @constraints.dependent_property
    # def support(self):
    #     result = self.base_dist.support
    #     if self.reinterpreted_batch_ndims:
    #         result = constraints.independent(result, self.reinterpreted_batch_ndims)
    #     return result

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def variance(self):
        return self.base_dist.variance

    def sample(self, sample_shape=()):
        return self.base_dist.sample(sample_shape)

    def rsample(self, sample_shape=()):
        return self.base_dist.rsample(sample_shape)

    def log_prob(self, value):
        log_prob = self.base_dist.log_prob(value)
        x = sum_rightmost(log_prob, self.reinterpreted_batch_ndims)
        return x

    def entropy(self):
        entropy = self.base_dist.entropy()
        return sum_rightmost(entropy, self.reinterpreted_batch_ndims)

    def enumerate_support(self, expand=True):
        if self.reinterpreted_batch_ndims > 0:
            raise NotImplementedError("Enumeration over cartesian product is not implemented")
        return self.base_dist.enumerate_support(expand=expand)

    def __repr__(self):
        return self.__class__.__name__ + '({}, {})'.format(self.base_dist, self.reinterpreted_batch_ndims)


class Categorical(Distribution):
    r"""
    Creates a categorical distribution parameterized by either :attr:`probs` or
    :attr:`logits` (but not both).

    .. note::
        It is equivalent to the distribution that :func:`torch.multinomial`
        samples from.

    Samples are integers from :math:`\{0, \ldots, K-1\}` where `K` is ``probs.size(-1)``.

    If `probs` is 1-dimensional with length-`K`, each element is the relative probability
    of sampling the class at that index.

    If `probs` is N-dimensional, the first N-1 dimensions are treated as a batch of
    relative probability vectors.

    .. note:: The `probs` argument must be non-negative, finite and have a non-zero sum,
              and it will be normalized to sum to 1 along the last dimension. :attr:`probs`
              will return this normalized value.
              The `logits` argument will be interpreted as unnormalized log probabilities
              and can therefore be any real number. It will likewise be normalized so that
              the resulting probabilities sum to 1 along the last dimension. :attr:`logits`
              will return this normalized value.

    See also: :func:`torch.multinomial`

    Example::

        >>> m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        >>> m.sample()  # equal probability of 0, 1, 2, 3
        tensor(3)

    Args:
        probs (Tensor): event probabilities
        logits (Tensor): event log probabilities (unnormalized)
    """
    arg_constraints = {'probs': simplex,
                       'logits': real_vector}
    has_enumerate_support = True

    def __init__(self, probs=None, validate_args=None):
        assert probs is not None
        if probs is not None:
            if len(probs.shape) < 1:
                raise ValueError("`probs` parameter must be at least one-dimensional.")
            self.probs = probs / probs.sum(-1, keepdims=True)
        self._param = self.probs
        self._num_events = self._param.shape[-1]
        batch_shape = self._param.shape[:-1] if len(self._param.shape) > 1 else ()
        super(Categorical, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Categorical, _instance)
        param_shape = batch_shape + (self._num_events,)
        if 'probs' in self.__dict__:
            new.probs = self.probs.expand(param_shape)
            new._param = new.probs
        new._num_events = self._num_events
        super(Categorical, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    # @constraints.dependent_property(is_discrete=True, event_dim=0)
    # def support(self):
    #     return constraints.integer_interval(0, self._num_events - 1)

    # @lazy_property
    # def logits(self):
    #     return probs_to_logits(self.probs)

    # @lazy_property
    # def probs(self):
    #     return logits_to_probs(self.logits)

    @property
    def param_shape(self):
        return self._param.shape

    @property
    def mean(self):
        return F.full(self._extended_shape(), float("nan"), 
                      dtype=self.probs.dtype)

    @property
    def variance(self):
        return F.full(self._extended_shape(), float("nan"), 
                      dtype=self.probs.dtype)

    def sample(self, sample_shape=()):
        probs_2d = self.probs.reshape(-1, self._num_events)
        samples_2d = F.transpose(multinomial_sample(probs_2d, get_numel(sample_shape), True), (1, 0))
        return samples_2d.reshape(self._extended_shape(sample_shape))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # print(value.shape) # N_data x 1
        value = F.expand_dims(value.astype('int32'), axis=-1)
        # print(value.shape) # N_data x 1
        logits = probs_to_logits(self.probs)
        # print(logits.shape) # N_data x D_space
        assert len(logits.shape) == 2 and len(value.shape) == 2
        assert logits.shape[0] == value.shape[0] and value.shape[1] == 1
        target_shape = (value.shape[0], logits.shape[1])
        value = F.broadcast_to(value, target_shape)
        # N_data x D_space
        value = value[..., :1]
        return F.squeeze(F.gather(logits, -1, value), axis=-1)

    def entropy(self):
        logits = probs_to_logits(self.probs)
        min_real = np.finfo(np.float32).min.item()
        safe_logits = F.clip(logits, lower=min_real)
        p_log_p = safe_logits * self.probs
        return -p_log_p.sum(-1)

    # def enumerate_support(self, expand=True):
    #     num_events = self._num_events
    #     values = F.arange(num_events, dtype="long")
    #     values = values.view((-1,) + (1,) * len(self._batch_shape))
    #     if expand:
    #         values = values.expand((-1,) + self._batch_shape)
    #     return values


class ExponentialFamily(Distribution):
    r"""
    ExponentialFamily is the abstract base class for probability distributions belonging to an
    exponential family, whose probability mass/density function has the form is defined below

    .. math::

        p_{F}(x; \theta) = \exp(\langle t(x), \theta\rangle - F(\theta) + k(x))

    where :math:`\theta` denotes the natural parameters, :math:`t(x)` denotes the sufficient statistic,
    :math:`F(\theta)` is the log normalizer function for a given family and :math:`k(x)` is the carrier
    measure.

    Note:
        This class is an intermediary between the `Distribution` class and distributions which belong
        to an exponential family mainly to check the correctness of the `.entropy()` and analytic KL
        divergence methods. We use this class to compute the entropy and KL divergence using the AD
        framework and Bregman divergences (courtesy of: Frank Nielsen and Richard Nock, Entropies and
        Cross-entropies of Exponential Families).
    """

    @property
    def _natural_params(self):
        """
        Abstract method for natural parameters. Returns a tuple of Tensors based
        on the distribution
        """
        raise NotImplementedError

    def _log_normalizer(self, *natural_params):
        """
        Abstract method for log normalizer function. Returns a log normalizer based on
        the distribution and input
        """
        raise NotImplementedError

    @property
    def _mean_carrier_measure(self):
        """
        Abstract method for expected carrier measure, which is required for computing
        entropy.
        """
        raise NotImplementedError

    # def entropy(self):
    #     """
    #     Method to compute the entropy using Bregman divergence of the log normalizer.
    #     """
    #     result = -self._mean_carrier_measure
    #     nparams = [p.detach().requires_grad_() for p in self._natural_params] 
    #     lg_normal = self._log_normalizer(*nparams)
    #     gradients = torch.autograd.grad(lg_normal.sum(), nparams, create_graph=True)
    #     result += lg_normal
    #     for np, g in zip(nparams, gradients):
    #         result -= np * g
    #     return result


class Normal(ExponentialFamily):
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by
    :attr:`loc` and :attr:`scale`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterinistic")
        >>> m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # normally distributed with loc=0 and scale=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution
            (often referred to as sigma)
    """
    arg_constraints = {'loc': real, 'scale': positive}
    support = real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return F.pow(self.stddev, 2)

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = loc, scale
        assert isinstance(loc, mge.Tensor)
        assert loc.shape == scale.shape
        batch_shape = self.loc.shape
        super(Normal, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Normal, _instance)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Normal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=()):
        shape = self._extended_shape(sample_shape)
        return mge.random.normal(size=shape) * self.scale + self.loc

    def rsample(self, sample_shape=()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype)
        return self.loc + eps * self.scale

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = (self.scale ** 2)
        log_scale = F.log(self.scale)
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

    # def cdf(self, value):
    #     if self._validate_args:
    #         self._validate_sample(value)
    #     return 0.5 * (1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2)))

    # def icdf(self, value):
    #     return self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + F.log(self.scale)

    @property
    def _natural_params(self):
        return (self.loc / F.pow(self.scale, 2), -0.5 * 1 / F.pow(self.scale, 2))

    def _log_normalizer(self, x, y):
        return -0.25 * F.pow(x, 2) / y + 0.5 * F.log(-math.pi / y)


def multinomial_sample(weights, num_sample, replacement=False):
    assert replacement
    assert len(weights.shape) == 2
    assert num_sample == 1
    assert (weights < 0.).sum() == 0
    
    weights_s = F.cumsum(weights, axis=-1)
    r = mge.random.uniform(size=(len(weights), 1))
    m = (weights_s < r)
    res = F.clip(m.sum(axis=-1, keepdims=True), lower=0, upper=weights.shape[1]-1)
    
    assert (res >= weights.shape[1]).sum() == 0
    return res


def get_numel(t):
    return reduce(lambda x,y:x*y, t, 1)


def clamp_probs(probs):
    eps = 1e-8
    return F.clip(probs, lower=eps, upper=1-eps)

def probs_to_logits(probs, is_binary=False):
    r"""
    Converts a tensor of probabilities into logits. For the binary case,
    this denotes the probability of occurrence of the event indexed by `1`.
    For the multi-dimensional case, the values along the last dimension
    denote the probabilities of occurrence of each of the events.
    """
    ps_clamped = clamp_probs(probs)
    if is_binary:
        return F.log(ps_clamped) - F.log1p(-ps_clamped)
    return F.log(ps_clamped)




def _standard_normal(shape, dtype):
    return mge.random.normal(size=shape)