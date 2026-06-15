# (c) 2024-04-03 Martin Matousek
# Last change: $Date$
#              $Revision$
import random
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import scipy.sparse as sp

_EPS = np.finfo(float).eps
_EPS_SQ = np.sqrt(_EPS)

class IOptimLogger(ABC):
    """Interface for loggers used by optimization functions."""

    @abstractmethod
    def start(self, label: str) -> None:
        """Message for an optimization start."""
        pass

    @abstractmethod
    def iteration(self, i: int, msg: str) -> None:
        """ Message for an iteration that has not improved the solution."""
        pass

    @abstractmethod
    def good_iteration(self, i: int, msg: str) -> None:
        """ Message for an iteration that has improved the solution."""
        pass

    @abstractmethod
    def summary(self, msg: str) -> None:
        """
        Message for an optimization summary (can be called multiple times).
        """
        pass

    @abstractmethod
    def finish(self, niter: int) -> None:
        """Message for an optimization finish."""

class _OptimLoggerDefault(IOptimLogger):
    """Default console logger for optimization functions."""

    def __init__(self, level: str):
        """Constructor.

        :param level: verbosity level: 'none', 'summary', 'good-iter', 'iter'
        """

        if level == 'none':
            self.is_summary = False
            self.is_iter = False
            self.is_good_iter = False
        elif level == 'summary':
            self.is_summary = True
            self.is_good_iter = False
            self.is_iter = False
        elif level == 'good-iter':
            self.is_summary = True
            self.is_good_iter = True
            self.is_iter = False
        elif level == 'iter':
            self.is_summary = True
            self.is_good_iter = True
            self.is_iter = True
        else:
            raise ValueError('Wrong verbosity level.')

        self.label = ''

    def start(self, label: str) -> None:
        self.label = label
        if self.is_summary:
            print('%s ...' % label)

    def summary(self, msg: str) -> None:
        if self.is_summary:
            print('  ' + msg)


    def finish(self, niter: int) -> None:
        if self.is_summary:
            print('%s finish: Iterations = %i\n' % (self.label, niter))

    def good_iteration(self, i: int, msg: str) -> None:
        if self.is_good_iter:
            print('  Iter % 3i: %s' % (i, msg))

    def iteration(self, i: int, msg: str) -> None:
        if self.is_iter:
            print('  Iter % 3i: %s' % (i, msg))


def _determine_optim_logger(verbose: Union[str, IOptimLogger]) -> IOptimLogger:
    """Prepare a logger object for optimization functions."""

    if isinstance(verbose, str):
        return _OptimLoggerDefault(verbose)
    elif isinstance(verbose, IOptimLogger):
        return verbose
    else:
        raise ValueError('Unhandled verbosity argument')


def fmin_lsq(x0, efun, delta=1e-4, **kwargs ):

    dim = x0.shape[0]
    num = efun(x0).reshape(-1).shape[0]

    def fun(x):
        return (efun(x).reshape(-1)**2).sum()

    def e_jaco(x):
        e = efun(x).reshape(-1)

        J = np.zeros((num, dim))
        for i in range(dim):
            x_ = x.copy()
            x_[i] += delta
            e_ = efun(x_).reshape(-1)
            J[:, i] = (e_ - e) / delta

        return e, J

    def gHfun(x):
        e, J = e_jaco(x)
        Jt = J.T
        g = 2 * Jt @ e
        H = 2 * (Jt @ J)

        return g, H

    retval = levmar(x0, fun, gHfun, **kwargs)
    return retval


def levmar(x0, fun, gh_fun,
           mindf: float = _EPS_SQ,
           ming: float = _EPS_SQ,
           maxiter: int = 1000000,
           maxfeval: int = 10000000,
           mu: float = 1.0,
           maxmu: float = 1 / _EPS,
           cfun = None,
           xfun = None,
           verbose = 'iter',
           stats: bool = False):
    """Second order minimization by Levenberg-Maquardt method.

       [x, stat] = levmar(x0, fun, gHfun, ...)

    Objective function is minimized by Levenberq-Maquardt algorithm
    (a damped Gauss-Newton method).


         x0    .. initial estimate (column vector)

         fun   .. function returning criterion value for a parameter vector;
                  called as f = fun(x)

         gHfun .. function returning gradient vector and Hessian matrix for a
                  parameter vector; called as [g, H] = gHfun(x)

       Optional named arguments (defaults in parentheses):
         'mindf'    .. stopping criterion: minimum change of criterion value
                       in each iteration (sqrt(eps))

         'ming'     .. stopping criterion: minimum gradient length (sqrt(eps))

         'maxiter'  .. maximum number of iterations (1e6)

         'maxfeval' .. maximum number of criterion function calls (1e7)

         'mu'       .. initial value of damping parameter (1)

         'maxmu'    .. stopping criterion: maximum value of mu (1/eps)

         'cfun'     .. constraint function and its derivation ([]); called
                       as [c, c_der] = cfun(x)

         'xfun'     .. function modifying x; x = xfun(x) is called after
                       each update; can be e.g. used to normalize gauge freedom ([])

         'verbose'  .. 'none', 'summary', 'good-iter', 'iter' or
                        a logger object ('iter')

         'stats'    .. if true, detailed optimisation stats are returned (false)

       Output:
         x     .. the found optimum

         stat  .. optimisation stats, struct with fields:
           .iter    .. number of iterations
           .numeval .. number of criterion function calls
           .x0      .. the initial estimate
           .x       .. the found optimum
           .f0      .. criterion (fun) value at x0
           .f       .. criterion (fun) value at x
           .t       .. elapsed time
           The following fields are produced only when 'stats' is turned on:
           .X       .. values of x for every iteration
           .F       .. values of criterion for every iteration
           .MU      .. values of damping factor for every iteration
    """

    # prepare options and variables

    #t = tic;

    mingq = ming**2  # we compute gradient length squared

    logger = _determine_optim_logger(verbose)

    logger.start('LEVMAR')

    x = x0           # parameter vector
    n = len(x0)   # dimension of parameter vcector
    iter = 0         # iteration counter
    fiter = 0        # function call counter
    f = fun(x)       # criterion value
    f0 = f

    #if size(x, 2) ~= 1 || numel(size(x)) > 2
    #    error('Initial parameter vector x0 must be a column vector');

    #if numel(f) ~= 1
    #    error('The criterion fun(x) must be a scalar function');

    # identity matrix of proper size and optional sparsity
    _, h, = gh_fun(x)

    if sp.issparse(h):
        is_sparse = True
        identity = sp.eye(n, n)
    else:
        is_sparse = False
        identity = np.eye(n, n)

    if stats: # saving of trajectories
        x_list = []
        f_list = []
        mu_list = [mu]

    logger.good_iteration(0, '[]          f=%8.3g' % f)

    # iteration

    do_break = False
    c = None  # constraints
    while True:

        # gradient and hessian
        [g, h] = gh_fun(x)

        # constraints
        if cfun is not None:
            c, C = cfun(x)
            N = np.zeros(c.shape[0]) # TODO move up

        # stopping criteria
        if iter > maxiter:
            #warning('LEVMAR:MaxIterReached', 'Maximum iteration count reached');
            logger.summary('terminate: maximum iteration count reached (%i)' %
                           iter)
            break

        glenq = g @ g
        if glenq < mingq:
            logger.summary('terminate: gradient below threshold (%g < %g)' %
                           (np.sqrt(glenq), ming))
            break

        iter += 1

        # Do a step, change mu until objective function is not decreased
        while True:
            if mu > maxmu:
                #warning('LEVMAR:MaxMiReached', 'Maximum mu reached');
                logger.summary('terminate: maximum mu reached (2^%f)' %
                               np.log2(mu))
                do_break = True
                break

            # damped H
            hdamp = h + mu * identity

            # compute parameter vector change
            if c is None:  # no constraints, solve Hdamp * dx = -g
                if is_sparse:
                    dx = sp.linalg.spsolve(hdamp, -g)
                else:
                    dx = np.linalg.solve(hdamp, -g)

            else:  # some constraints, use Sequential Quadratic Programming
                pass # TODO
                #hh = [hdamp, C'; C, N];
                #dxl = -hh \ [g; c];
                #dx = dxl(1:n);

            x_new = x + dx

            # optional modification of x
            if xfun is not None:
                x_new = xfun(x_new)

            f_new = fun(x_new)
            fiter += 1

            if fiter > maxfeval:
                #warning('LEVMAR:MaxFevalReached', 'Maximum feval count reached');
                logger.summary('terminate: maximum feval count reached (%i)' %
                               fiter)
                do_break = 1
                break

            # Good turn: break if criterion not increased; test must be '<=',
            # not '<', sometimes |g| > 0 and f = f_new due to numeric accuracy.
            if f_new <= f:
                break

            # Bad turn: strengthen the gradient part, shorten the step.
            mu *= 2
            logger.iteration(iter, '(^) mu=2^%i' % np.round(np.log2(mu)))

        if do_break:
            break

        # now f_new <= f

        # Check local accuracy of the second order approximation - compare the
        # estimated, and the real decrease of objective function and modify
        # the mu accordingly.

        df_est = g @ dx + 0.5 * dx @ h @ dx
        df = f_new - f

        ratio = df / df_est

        if ratio < 0.25:  # wrong approximation, strengthen the gradient step
            mu *= 2
            mutx = '[^] mu=2^%i' % np.round(np.log2(mu))
        elif ratio > 0.75:  # good approximation, strengthen the Newton step
            mu /= 2
            mutx = '[v] mu=2^%i' % np.round(np.log2(mu))
        else: # keep as is
            mutx = '[]'

        # update f, x
        f = f_new
        x = x_new

        logger.good_iteration(iter,
                              '%-12s f=%8.3g df=%9.4g |g|=%9.4g |dx|=%8.3g' %
                              (mutx, f, -df, np.sqrt(glenq), np.sqrt(dx @ dx)))

        if stats:
            x_list += [x]
            f_list += [f]
            mu_list += [mu]

        if np.abs(df) < mindf:
            logger.summary(
                'terminate: criterion change below threshold (%g < %g)' %
                (np.abs(df), mindf))
            break

    # finish
    #t = toc(t);
    t = 0  # TODO

    stat = {
        'iter': iter,
        'numeval': fiter,
        'f0': f0,
        'f': f,
        'x0': x0,
        'x': x,
        't': t
    }

    if stats:
        stat['x_list'] = x_list
        stat['f_list'] = f_list
        stat['mu_list'] = mu_list

    logger.summary('%i funcalls in %g sec, f %g->%g' % (fiter, t, f0, f))
    logger.finish(iter)

    return x


def ransac(num_points, mss, log_prob, support_fcn,
           verbose='iter',
           maxiter=100000,
           inl_min=0):

    logger = _determine_optim_logger(verbose)

    N_max = np.inf
    if inl_min:
        w = inl_min / num_points
        N_max = np.ceil(- log_prob / np.log10(1 - w**mss))
        print(N_max)

    iter = 0
    best_support = -np.inf
    best_num_inl = 0
    best_sample = None

    logger.start('RANSAC')

    rng = np.random.default_rng()
    while True:
        iter += 1

        if iter >= N_max:
            logger.summary('terminate: stopping criterion reached')
            break

        if iter >= maxiter:
            #warning('RANSAC:MaxIterReached', 'Maximum iteration count reached');
            logger.summary('terminate: maximum iteration count reached (%i)' %
                           iter)
            break

        sample = rng.choice(num_points, mss, replace=False)

        s = support_fcn(sample, best_support)

        num_inl, support = s[0], s[1]  # s can have additional values

        if num_inl < mss:
            # it can happen that no model can be computed from the sample
            logger.iteration(iter, 'support = none (%i of %i)' % (num_inl,
                                                                  num_points))
            continue

        if support > best_support:
            best_support = support
            best_sample = sample
            best_num_inl = num_inl

            # stopping criterion based on currently best inlier ratio
            if num_inl == num_points:
                N_max = 0
            else:
                w = num_inl / num_points
                N_max = np.ceil(- log_prob / np.log10(1 - w**mss))

            logger.good_iteration(iter, 'support = %f (%i of %i), N_max = %i' %
                                    (support, num_inl, num_points, N_max))
        else:
            logger.iteration(iter, 'support = %f (%i of %i)' %
                             (support, num_inl, num_points))


    logger.summary('support = %f (%i of %i)' % (best_support, best_num_inl,
                    num_points))
    logger.finish(iter)


    return best_sample