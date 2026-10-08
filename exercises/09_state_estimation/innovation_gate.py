"""Track 09 - State estimation.

A robot never sees its own state; it sees sensors. The gyro is smooth but
drifts, the accelerometer is noisy but does not drift, and the Kalman filter
is the general way to weigh a prediction against a measurement. These
exercises build the pieces one at a time.
"""

import numpy as np


def innovation_gate(z, z_pred, S, threshold):
    """Should a measurement be used at all?

    A glitched sensor, a wrong data association, a reflection: the filter
    has no way to know, except that the measurement disagrees with its own
    prediction by more than the prediction's uncertainty allows. The
    squared Mahalanobis distance of the innovation measures exactly that::

        y = z - z_pred
        d2 = y.T @ inv(S) @ y

    Accept the measurement when ``d2 <= threshold``. The threshold comes
    from the chi-squared distribution with ``m`` degrees of freedom (for
    ``m = 2`` and 99 %, about ``9.21``), the caller supplies it.

    Use ``np.linalg.solve``, not ``inv``, and use the full ``S``: taking
    only its diagonal ignores correlated measurement axes and accepts
    things it should not.

    ``z`` and ``z_pred`` are ``(m,)``, ``S`` is ``(m, m)``.

    Returns ``(accept, d2)``: a Python ``bool`` and a Python ``float``.
    """
    raise NotImplementedError("TODO: write this function")
