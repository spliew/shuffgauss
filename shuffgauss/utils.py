import math

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm

__all__ = ["logstirling", "fun_int", "accel_asc"]


def logstirling(l: int, k: int):
    """Calculate large binomial coefficients with stirling approx.
    for large k, see
    https://math.stackexchange.com/questions/64716/approximating-the-logarithm-of-the-binomial-coefficient
    for small k (sqrt(n)), see
    https://math.stackexchange.com/questions/1447296/stirlings-approximation-for-binomial-coefficient
    """
    if k < np.sqrt(l):
        return k * np.log(l) - gammaln(k + 1)
    else:
        return l * np.log(l) - k * np.log(k) - (l - k) * np.log(l - k)


def fun_int(i: int, delta: float, coeff: int, RDPs_int: np.ndarray):
    """Relates RDP to approximate DP.

    Parameters
    -----------
    RDPs_int: array-like, shape (max lambda)
        RDP starting with \lambda =1

    i is lmbda
    """
    if i <= 1 or i > RDPs_int.shape[0]:
        return np.inf
    else:
        return (np.log(1 / delta) + (i - 1) * np.log(1 - 1 / i) - np.log(i)) / (
            i - 1
        ) + RDPs_int[i - 1] * coeff


def inv_permute(k: list, n: int):
    """Calculate the permutation invariant part of multinomial expansion.
    given eg, x_1^2x_2^2x_3^1, return the no. of permutation w.p. to
    the underscore index
    """
    zeros = n - len(k)
    k = np.array(k)
    _, counts = np.unique(k, return_counts=True)
    if zeros >= 0:
        return gammaln(n + 1) - (gammaln(zeros + 1) + gammaln(counts + 1).sum())
    else:
        return gammaln(n + 1) - (gammaln(counts + 1).sum())


# multinomial coeff
def eff_log_multinomial_coeff(k, n):
    """Given a partition k of n,
    return the multinomial coefficient times the
    no of invariant permutation
    """
    return log_multinomial_coeff(k) + inv_permute(k, n)


def log_multinomial_coeff(c):
    """Calculate the log of multinomial coefficients
    :param list of number:
    :return: multinomial coeff
    """
    c = np.array(c)
    return gammaln(c.sum() + 1) - gammaln(c + 1).sum()


# fast partition. from https://jeromekelleher.net/generating-integer-partitions.html


def accel_asc(n):
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[: k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[: k + 1]


def stable_logsumexp(x):
    a = np.max(x)
    return a + np.log(np.sum(np.exp(x - a)))


def stable_logsumexp_two(x, y):
    a = np.maximum(x, y)
    if np.isneginf(a):
        return a
    else:
        return a + np.log(np.exp(x - a) + np.exp(y - a))


def logcomb(n, k):

    return gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1)


def get_binom_coeffs(sz):
    C = np.zeros(shape=(sz + 1, sz + 1))
    # for k in range(1,sz + 1,1):
    #    C[0, k] = -np.inf
    for n in range(sz + 1):
        C[n, 0] = 0  # 1
    for n in range(1, sz + 1, 1):
        C[n, n] = 0
    for n in range(1, sz + 1, 1):
        for k in range(1, n, 1):
            # numerical stable way of implementing the recursion rule
            C[n, k] = stable_logsumexp_two(C[n - 1, k - 1], C[n - 1, k])
    # only the lower triangular part of the matrix matters
    return C


def get_binom_coeffs_dict(sz):
    C = {}  # np.zeros(shape = (sz + 1, sz + 1));
    # for k in range(1,sz + 1,1):
    #    C[0, k] = -np.inf
    for n in range(sz + 1):
        C[(n, 0)] = 0  # 1
    for n in range(1, sz + 1, 1):
        C[(n, n)] = 0
    for n in range(1, sz + 1, 1):
        for k in range(1, n, 1):
            # numerical stable way of implementing the recursion rule
            C[(n, k)] = stable_logsumexp_two(C[(n - 1, k - 1)], C[(n - 1, k)])
    # only the lower triangular part of the matrix matters
    return C


def expand_binom_coeffs_dict(C, sz, sznew):
    for n in range(sz, sznew + 1, 1):
        C[(n, 0)] = 0
    for n in range(sz, sznew + 1, 1):
        C[(n, n)] = 0
    for n in range(sz, sznew + 1, 1):
        for k in range(1, n, 1):
            C[(n, k)] = stable_logsumexp_two(C[(n - 1, k - 1)], C[(n - 1, k)])
    return C  # also need to update the size of C to sznew whenever this function is called just to keep track.
