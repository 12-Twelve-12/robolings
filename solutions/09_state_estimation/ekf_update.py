"""Track 09 - State estimation.

A robot never sees its own state; it sees sensors. The gyro is smooth but
drifts, the accelerometer is noisy but does not drift, and the Kalman filter
is the general way to weigh a prediction against a measurement. These
exercises build the pieces one at a time.
"""

import numpy as np


def ekf_update(x, P, z, h, H, R):
    """The update half of an extended Kalman filter.

    Compare the measurement with what the state predicts, and move the state
    towards it by the Kalman gain::

        y = z - h(x)                      innovation
        S = H @ P @ H.T + R               innovation covariance
        K = P @ H.T @ inv(S)              gain
        x_new = x + K @ y
        P_new = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T

    The last line is the Joseph form. The textbook ``(I - K H) P`` is
    algebraically the same but loses symmetry and positive-definiteness
    after a few thousand steps in floating point, so use the long one.

    ``h`` is a Python callable, ``h(x)`` returns the predicted measurement
    ``(m,)``. ``H`` is its ``(m, n)`` Jacobian at ``x``. ``R`` is the
    ``(m, m)`` measurement noise. Use ``np.linalg.solve`` rather than
    forming ``inv(S)``.

    Returns ``(x_new, P_new)``.
    """
    # >>> solution
    x = np.asarray(x, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    y = z - np.asarray(h(x), dtype=np.float64)
    S = H @ P @ H.T + R
    K = np.linalg.solve(S, H @ P).T  # P H^T S^-1, S symmetric
    x_new = x + K @ y
    IKH = np.eye(len(x)) - K @ H
    P_new = IKH @ P @ IKH.T + K @ R @ K.T
    return x_new, P_new
    # <<< solution
