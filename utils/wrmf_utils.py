"""
 original source code : https://github.com/PreferredAI/cornac/blob/master/cornac/utils/common.py
"""


import numpy as np
import numbers
from utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
)


class CornacException(Exception):
    """Exception base class to extend from
    Original source code - "https://github.com/PreferredAI/cornac/blob/master/cornac/utils/common.py"
    """

    pass


class ScoreException(CornacException):
    """Exception raised in score function when facing unknowns

    """

    pass


def clip_by_bound(values, lower_bound, upper_bound):
    """enforce values to lie in a [lower_bound, upper_bound] range

    Args:
        values (np.array): values to be clipped.
        lower_bound (scalar): Lower bound.
        upper_bound (scalar): Upper bound.

    Returns:
        np.array: Clipped values in range [lower_bound, upper_bound]
    """
    values = np.where(values > upper_bound, upper_bound, values)
    values = np.where(values < lower_bound, lower_bound, values)

    return values


def get_rng(seed):
    """Return a RandomState of Numpy.
    Original source code - "https://github.com/PreferredAI/cornac/blob/master/cornac/utils/common.py"
    If seed is None, use RandomState singleton from numpy.
    If seed is an integer, create a RandomState from that seed.
    If seed is already a RandomState, just return it.
    """

    if seed is None:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('{} can not be used to create a numpy.random.RandomState'.format(seed))


def uniform(shape=None, low=0.0, high=1.0, random_state=None, dtype=np.float32):
    """Draw samples from a uniform distribution.
    Original source code - "https://github.com/PreferredAI/cornac/blob/master/cornac/utils/init_utils.py"
    Args:
        shape (int or tuple of ints): Output shape. If shape is ``None`` (default), a single value is returned.
        low (float or array_like of floats): Lower boundary of the output interval. The default value is 0.
        high (float or array_like of floats):  Upper boundary of the output interval. The default value is 1.0.
        random_state (int or np.random.RandomState): If an integer is given, it will be used as seed value for creating a RandomState.
        dtype (str or dtype): Returned data-type for the output array.

    Returns:
        ndarray or scalar: Drawn samples from the parameterized uniform distribution.
    """
    return get_rng(random_state).uniform(low, high, shape).astype(dtype)


def xavier_uniform(shape, random_state=None, dtype=np.float32):
    """Return a numpy array by performing 'Xavier' initializer also known as 'Glorot' initializer on Uniform distribution.
    Original source code - "https://github.com/PreferredAI/cornac/blob/master/cornac/utils/init_utils.py"
    Args:
        shape (int or tuple of ints): Output shape.
        random_state (int or np.random.RandomState): If an integer is given, it will be used as seed value for creating a RandomState.
        dtype (str or dtype): Returned data-type for the output array.

    Returns:
        ndarray: Output matrix.
    """

    assert len(shape) == 2  # only support matrix
    std = np.sqrt(2.0 / np.sum(shape))
    limit = np.sqrt(3.0) * std
    return uniform(shape, -limit, limit, random_state, dtype)


def weight_user_oriented(data, alpha):
    """Return weight vector based on user-oriented strategy (Pan, Rong, et al. One-class collaborative filtering. 2008)

    Args:
        data (pd.DataFrame): Implicit Feedback DataFrame which is binarized.
        alpha (scalar): Hyper-parameter that controls the strength of weights

    Returns:
        np.array: (n_users, ) size weight array
    """
    return data[DEFAULT_USER_COL].value_counts().sort_index().to_numpy() * alpha


def weight_item_oriented(data, alpha):
    """Return weight vector based on item-oriented strategy (Pan, Rong, et al. One-class collaborative filtering. 2008)

    Args:
        data (pd.DataFrame): Implicit Feedback DataFrame which is binarized.
        alpha (scalar): Hyper-parameter that controls the strength of weights

    Returns:
        np.array: (n_items, ) size weight array
    """
    n_users = data[DEFAULT_USER_COL].nunique()
    return alpha * (n_users - data[DEFAULT_ITEM_COL].value_counts().sort_index().to_numpy())


def weight_item_popularity(data, alpha, c_0):
    """Return weight vector based on item-popularity strategy
        (He, Xiangnan, et al. Fast matrix factorization for online recommendation with implicit feedback. 2016)

    Args:
        data (pd.DataFrame): Implicit Feedback DataFrame which is binarized.
        alpha (scalar): Hyper-parameter that controls the significance level of popular items over unpopular ones.
            If alpha > 1,  the difference of weights between popular items and unpopular ones is strengthened.
            If 0 < alpha < 1 the difference is weakened and the weight of popular items is suppressed.
        c_0: Hyper-parameter that determines the overall weight of unobserved instances.

    Returns:
         np.array: (n_items, ) size weight array
    """
    u_j = data[DEFAULT_ITEM_COL].value_counts().sort_index().to_numpy()
    f_vec = u_j / u_j.sum()
    f_alpha_vec = f_vec ** alpha
    return c_0 * (f_alpha_vec / f_alpha_vec.sum())
