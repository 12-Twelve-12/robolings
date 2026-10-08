import numpy as np

from helpers import axis_angle_quat, same_rotation_quat
from loader import load_track

m = load_track("09_state_estimation")
RNG = np.random.default_rng(9)  # rebound before every test, see conftest.py


def quat_to_matrix(q):
    """Reference conversion, for the tests only."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


class TestComplementaryFilter:
    def test_pure_gyro_integrates(self):
        out = m.complementary_filter(0.1, 0.5, 99.0, 0.01, 1.0)
        assert isinstance(out, float)
        assert np.isclose(out, 0.1 + 0.5 * 0.01)

    def test_pure_accelerometer_copies_the_measurement(self):
        assert np.isclose(m.complementary_filter(0.1, 0.5, 0.3, 0.01, 0.0), 0.3)

    def test_blend(self):
        out = m.complementary_filter(1.0, 2.0, 0.0, 0.1, 0.9)
        assert np.isclose(out, 0.9 * 1.2 + 0.1 * 0.0)

    def test_gyro_integrates_from_the_previous_estimate(self):
        # if it integrated from the accelerometer angle instead, this would be 0.98 * 0.5 + 0.02 * 0.5
        out = m.complementary_filter(0.0, 1.0, 0.5, 0.5, 0.98)
        assert np.isclose(out, 0.98 * 0.5 + 0.02 * 0.5)
        out = m.complementary_filter(2.0, 1.0, 0.5, 0.5, 0.98)
        assert np.isclose(out, 0.98 * 2.5 + 0.02 * 0.5)

    def test_roll_and_pitch_at_once(self):
        out = m.complementary_filter([0.1, 0.2], [1.0, -1.0], [0.0, 0.0], 0.1, 0.5)
        assert out.shape == (2,)
        assert np.allclose(out, [0.1, 0.05])

    def test_drift_is_pulled_back(self):
        # constant true angle 0.3, gyro reads a bias, accelerometer is right on average
        angle = 0.0
        for _ in range(2000):
            angle = m.complementary_filter(angle, 0.05, 0.3 + RNG.normal(scale=0.05), 0.01, 0.98)
        assert abs(angle - 0.3) < 0.05


class TestGyroIntegrate:
    def test_zero_rate_leaves_q_alone(self):
        q = axis_angle_quat("z", 0.7)
        out = m.gyro_integrate(q, [0.0, 0.0, 0.0], 0.01)
        assert out.shape == (4,)
        assert np.allclose(out, q)

    def test_rotation_about_one_axis(self):
        out = m.gyro_integrate([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0], 0.25)
        assert same_rotation_quat(out, axis_angle_quat("z", 0.5))

    def test_body_frame_rate_composes_on_the_right(self):
        # start rotated 90 degrees about z; a body-x rate must appear as world-y rotation
        q0 = axis_angle_quat("z", np.pi / 2)
        out = m.gyro_integrate(q0, [1.0, 0.0, 0.0], 0.3)
        R = quat_to_matrix(out)
        expected = quat_to_matrix(q0) @ quat_to_matrix(axis_angle_quat("x", 0.3))
        assert np.allclose(R, expected)
        wrong = quat_to_matrix(axis_angle_quat("x", 0.3)) @ quat_to_matrix(q0)
        assert not np.allclose(R, wrong)

    def test_many_small_steps_match_one_big_one(self):
        omega = np.array([0.3, -0.2, 0.5])
        q = np.array([1.0, 0.0, 0.0, 0.0])
        for _ in range(100):
            q = m.gyro_integrate(q, omega, 0.01)
        one = m.gyro_integrate([1.0, 0.0, 0.0, 0.0], omega, 1.0)
        assert same_rotation_quat(q, one, atol=1e-9)

    def test_output_is_unit_length(self):
        q = RNG.normal(size=4)
        q /= np.linalg.norm(q)
        for _ in range(50):
            q = m.gyro_integrate(q, RNG.normal(size=3), 0.05)
        assert np.isclose(np.linalg.norm(q), 1.0)

    def test_tiny_rate_is_finite_and_right(self):
        out = m.gyro_integrate([1.0, 0.0, 0.0, 0.0], [1e-12, 0.0, 0.0], 0.01)
        assert np.all(np.isfinite(out))
        assert np.allclose(out, [1.0, 0.5e-14, 0.0, 0.0])
        out = m.gyro_integrate([1.0, 0.0, 0.0, 0.0], [1e-5, 0.0, 0.0], 0.01)
        assert np.allclose(out, axis_angle_quat("x", 1e-7))


class TestEkfPredict:
    def test_linear_model(self):
        F = np.array([[1.0, 0.1], [0.0, 1.0]])
        Q = np.diag([0.01, 0.02])
        P = np.diag([1.0, 2.0])
        x_pred, P_pred = m.ekf_predict([1.0, 3.0], P, lambda x: F @ x, F, Q)
        assert np.allclose(x_pred, [1.3, 3.0])
        assert np.allclose(P_pred, F @ P @ F.T + Q)

    def test_nonlinear_state_through_f(self):
        def f(x):
            return np.array([x[0] + 0.1 * np.cos(x[1]), x[1]])

        F = np.array([[1.0, -0.1 * np.sin(0.5)], [0.0, 1.0]])
        x_pred, _ = m.ekf_predict([0.0, 0.5], np.eye(2), f, F, np.zeros((2, 2)))
        assert np.allclose(x_pred, f(np.array([0.0, 0.5])))

    def test_covariance_grows_by_q(self):
        P = np.diag([0.5, 0.5])
        _, P_pred = m.ekf_predict([0.0, 0.0], P, lambda x: x, np.eye(2), 0.3 * np.eye(2))
        assert np.allclose(P_pred, 0.8 * np.eye(2))

    def test_input_covariance_is_not_modified(self):
        P = np.diag([0.5, 0.5])
        before = P.copy()
        m.ekf_predict([0.0, 0.0], P, lambda x: x, 2.0 * np.eye(2), np.eye(2))
        assert np.array_equal(P, before)

    def test_f_is_used_not_F(self):
        # F is only the Jacobian; the state goes through f
        F = np.array([[2.0, 0.0], [0.0, 2.0]])
        x_pred, _ = m.ekf_predict([1.0, 1.0], np.eye(2), lambda x: x + 1.0, F, np.zeros((2, 2)))
        assert np.allclose(x_pred, [2.0, 2.0])


class TestEkfUpdate:
    def setup_case(self):
        x = np.array([1.0, 2.0])
        P = np.array([[2.0, 0.5], [0.5, 1.0]])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.5]])
        return x, P, H, R

    def test_scalar_measurement_textbook_values(self):
        x, P, H, R = self.setup_case()
        x_new, P_new = m.ekf_update(x, P, [1.5], lambda s: H @ s, H, R)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        assert np.allclose(x_new, x + K @ (np.array([1.5]) - H @ x))
        IKH = np.eye(2) - K @ H
        assert np.allclose(P_new, IKH @ P @ IKH.T + K @ R @ K.T)

    def test_measurement_matching_the_prediction_changes_nothing_in_x(self):
        x, P, H, R = self.setup_case()
        x_new, _ = m.ekf_update(x, P, H @ x, lambda s: H @ s, H, R)
        assert np.allclose(x_new, x)

    def test_covariance_shrinks_and_stays_symmetric(self):
        x, P, H, R = self.setup_case()
        _, P_new = m.ekf_update(x, P, [1.5], lambda s: H @ s, H, R)
        assert np.allclose(P_new, P_new.T)
        assert P_new[0, 0] < P[0, 0]
        assert np.all(np.linalg.eigvalsh(P_new) > 0.0)

    def test_exact_measurement_pins_the_state(self):
        x, P, H, _ = self.setup_case()
        x_new, P_new = m.ekf_update(x, P, [7.0], lambda s: H @ s, H, np.array([[1e-12]]))
        assert np.isclose(x_new[0], 7.0, atol=1e-6)
        assert P_new[0, 0] < 1e-6

    def test_nonlinear_h_is_used(self):
        # h(x) = x0**2 is 9 here while H @ x is 18: a measurement of 9 must leave x alone
        x = np.array([3.0, 4.0])
        H = np.array([[6.0, 0.0]])  # Jacobian of x0**2 at x0 = 3
        x_new, _ = m.ekf_update(x, np.eye(2), [9.0], lambda s: np.array([s[0] ** 2]), H, np.array([[0.1]]))
        assert np.allclose(x_new, x)

    def test_joseph_form_keeps_an_ill_conditioned_update_symmetric(self):
        # thousands of updates with a nearly exact sensor: the short form drifts off symmetric
        x = np.array([0.0, 0.0])
        P = np.diag([1e6, 1e6])
        H = np.array([[1.0, 1e-3]])
        R = np.array([[1e-9]])
        for _ in range(500):
            x, P = m.ekf_update(x, P, [0.0], lambda s: H @ s, H, R)
            P = P + 1e-6 * np.eye(2)
        assert np.allclose(P, P.T, atol=1e-12, rtol=0.0)
        assert np.all(np.linalg.eigvalsh(P) > 0.0)


class TestInnovationGate:
    def test_matching_measurement_is_accepted_with_zero_distance(self):
        accept, d2 = m.innovation_gate([1.0, 2.0], [1.0, 2.0], np.eye(2), 9.21)
        assert isinstance(accept, bool)
        assert isinstance(d2, float)
        assert accept
        assert d2 == 0.0

    def test_distance_is_mahalanobis_not_euclidean(self):
        S = np.diag([4.0, 0.25])
        _, d2 = m.innovation_gate([2.0, 1.0], [0.0, 0.0], S, 100.0)
        assert np.isclose(d2, 2.0**2 / 4.0 + 1.0**2 / 0.25)

    def test_threshold_is_inclusive(self):
        accept, _ = m.innovation_gate([3.0], [0.0], [[1.0]], 9.0)
        assert accept
        accept, _ = m.innovation_gate([3.0], [0.0], [[1.0]], 8.99)
        assert not accept

    def test_correlated_axes_use_the_full_s(self):
        S = np.array([[1.0, 0.9], [0.9, 1.0]])
        # along the correlated direction the ellipse is long, across it short
        along, _ = m.innovation_gate([1.0, 1.0], [0.0, 0.0], S, 2.0)
        across, d2 = m.innovation_gate([1.0, -1.0], [0.0, 0.0], S, 2.0)
        assert along
        assert not across
        assert np.isclose(d2, 2.0 / 0.1)

    def test_an_outlier_is_rejected(self):
        accept, d2 = m.innovation_gate([10.0, 0.0], [0.0, 0.0], np.eye(2), 9.21)
        assert not accept
        assert np.isclose(d2, 100.0)
