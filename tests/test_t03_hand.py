import numpy as np

import t03_hand as m
from helpers import planar_finger

RNG = np.random.default_rng(3)


class TestEx10ExpandMimic:
    def test_coupled_finger(self):
        # Joints 0 and 1 are driven; 2 and 3 follow joint 1.
        q = m.expand_mimic([0.2, 0.6], 4, [0, 1], [(2, 1, 1.0, 0.0), (3, 1, 0.5, 0.1)])
        assert q.shape == (4,)
        assert np.allclose(q, [0.2, 0.6, 0.6, 0.4])

    def test_active_joints_in_any_order(self):
        q = m.expand_mimic([0.7, -0.3], 4, [3, 0], [(1, 3, 2.0, 0.0)])
        assert np.allclose(q, [-0.3, 1.4, 0.0, 0.7])

    def test_unlisted_joints_stay_at_zero(self):
        q = m.expand_mimic([1.0], 5, [2], [])
        assert np.allclose(q, [0.0, 0.0, 1.0, 0.0, 0.0])

    def test_negative_multiplier_and_offset(self):
        q = m.expand_mimic([0.5], 2, [0], [(1, 0, -1.5, 0.25)])
        assert np.allclose(q, [0.5, -0.5])


class TestEx11RetargetCost:
    def test_zero_at_a_perfect_match(self):
        human = RNG.normal(size=(5, 3))
        q = RNG.normal(size=6)
        assert m.retarget_cost(human, 1.2 * human, 1.2, q, q, 0.5) == 0.0

    def test_known_value(self):
        human = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        robot = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
        q = np.array([0.1, 0.2])
        q_prev = np.array([0.0, 0.4])
        # vector term: scale 1.5 -> diffs (0.5,0,0) and (0,2,-1) -> 0.5 * 5.25
        # smoothness: 0.5 * 2.0 * (0.01 + 0.04)
        cost = m.retarget_cost(human, robot, 1.5, q, q_prev, 2.0)
        assert np.isclose(cost, 2.625 + 0.05)

    def test_returns_a_python_float(self):
        cost = m.retarget_cost(np.ones((2, 3)), np.zeros((2, 3)), 1.0, np.zeros(3), np.zeros(3), 0.0)
        assert type(cost) is float
        assert np.isclose(cost, 3.0)

    def test_beta_weights_only_the_smoothness_term(self):
        human = RNG.normal(size=(4, 3))
        robot = RNG.normal(size=(4, 3))
        q = RNG.normal(size=5)
        q_prev = RNG.normal(size=5)
        base = m.retarget_cost(human, robot, 1.0, q, q_prev, 0.0)
        with_beta = m.retarget_cost(human, robot, 1.0, q, q_prev, 3.0)
        assert np.isclose(with_beta - base, 1.5 * np.sum((q - q_prev) ** 2))


class TestEx12FingertipIk:
    LOWER = np.array([-0.3, 0.0, 0.0])
    UPPER = np.array([1.5, 1.6, 1.4])

    def solve(self, target, q0, lower=None, upper=None, iters=200):
        chain = planar_finger()
        lower = self.LOWER if lower is None else lower
        upper = self.UPPER if upper is None else upper
        q = m.fingertip_ik(
            chain.tip,
            chain.numeric_position_jacobian,
            q0,
            target,
            lower,
            upper,
            iters,
            0.05,
        )
        return chain, np.asarray(q)

    def test_reaches_a_reachable_target(self):
        chain = planar_finger()
        target = chain.tip([0.4, 0.7, 0.5])
        chain, q = self.solve(target, [0.1, 0.1, 0.1])
        assert q.shape == (3,)
        assert np.linalg.norm(chain.tip(q) - target) < 1e-6

    def test_stays_inside_the_limits(self):
        chain = planar_finger()
        # Reachable only by bending backwards, which the limits forbid.
        target = chain.tip([-0.2, -0.8, -0.6])
        chain, q = self.solve(target, [0.2, 0.2, 0.2])
        assert np.all(q >= self.LOWER - 1e-12)
        assert np.all(q <= self.UPPER + 1e-12)

    def test_other_joints_compensate_for_a_joint_at_its_limit(self):
        chain = planar_finger()
        # Reachable inside the limits, but the unconstrained iteration drives
        # joint 0 past its upper limit of 0.05 rad.
        target = chain.tip([0.03, 1.2, 0.5])
        upper = np.array([0.05, 1.6, 1.4])
        chain, q = self.solve(target, [0.0, 0.1, 0.1], upper=upper, iters=200)
        assert q[0] <= 0.05 + 1e-12
        # Clipping only once at the end leaves an error of about 0.9 mm here.
        assert np.linalg.norm(chain.tip(q) - target) < 1e-6

    def test_does_not_modify_q0(self):
        chain = planar_finger()
        q0 = np.array([0.1, 0.1, 0.1])
        before = q0.copy()
        self.solve(chain.tip([0.3, 0.3, 0.3]), q0, iters=5)
        assert np.array_equal(q0, before)
