"""Track 05 - Imitation learning plumbing.

The data handling around a policy network. ACT, Diffusion Policy and the
current vision-language-action models all share these pieces, and bugs here
produce a policy that trains fine and then fails on the robot.
"""

import numpy as np


def dct_matrix(n):
    """Orthonormal DCT-II matrix of size ``n``.

    An action chunk is smooth in time, so most of its energy sits in a few
    low-frequency coefficients. Transforming before quantising is the idea
    behind the FAST action tokeniser: the high coefficients round to zero and
    the chunk becomes a short sequence of integers.

    NumPy has no DCT, so build the matrix::

        C[k, i] = s(k) * cos(pi * (2 i + 1) * k / (2 n))
        s(0) = sqrt(1 / n),  s(k > 0) = sqrt(2 / n)

    With those scale factors the matrix is orthonormal, so the inverse
    transform is just the transpose.

    Returns an ``(n, n)`` array.
    """
    # >>> solution
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    C = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    scale = np.full((n, 1), np.sqrt(2.0 / n))
    scale[0] = np.sqrt(1.0 / n)
    return scale * C
    # <<< solution


def tokenize(chunk, step):
    """Turn an action chunk into integer tokens.

    Transform along the time axis with ``dct_matrix``, then divide by ``step``
    and round to the nearest integer.

    ``chunk`` is ``(H, D)``; the transform is over ``H``, one column at a
    time, which ``C @ chunk`` already does.

    Returns an ``(H, D)`` integer array.
    """
    # >>> solution
    chunk = np.asarray(chunk, dtype=np.float64)
    coefficients = dct_matrix(chunk.shape[0]) @ chunk
    return np.rint(coefficients / step).astype(np.int64)
    # <<< solution


def detokenize(tokens, step):
    """Reconstruct an action chunk from its tokens.

    Multiply by ``step`` and transform back. Since ``dct_matrix`` is
    orthonormal, the inverse is its transpose, so no second matrix and no
    matrix inverse is needed.

    Returns an ``(H, D)`` float array.
    """
    # >>> solution
    tokens = np.asarray(tokens, dtype=np.float64)
    return dct_matrix(tokens.shape[0]).T @ (tokens * step)
    # <<< solution
