import numpy as np

from helpers import planar_finger
from loader import load_track

m = load_track("03_hand")
RNG = np.random.default_rng(3)


class TestExpandMimic:
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


class TestCoupledJacobian:
    def test_matches_the_chain_rule(self):
        # Joints 0 and 1 are driven, joint 2 follows joint 1 at half rate,
        # joint 3 is dead.
        J = RNG.normal(size=(3, 4))
        C = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.5], [0.0, 0.0]])
        out = m.coupled_jacobian(J, 4, [0, 1], [(2, 1, 0.5, 0.0)])
        assert out.shape == (3, 2)
        assert np.allclose(out, J @ C)

    def test_the_offset_is_a_constant_and_stays_out(self):
        J = RNG.normal(size=(3, 4))
        plain = m.coupled_jacobian(J, 4, [0, 1], [(2, 1, 0.5, 0.0)])
        shifted = m.coupled_jacobian(J, 4, [0, 1], [(2, 1, 0.5, 7.25)])
        assert np.allclose(plain, shifted)

    def test_source_indexes_the_full_vector_not_the_column(self):
        # The actuated joints are 3 and 0, in that order, so a joint that
        # follows joint 3 belongs in column 0.
        J = RNG.normal(size=(2, 4))
        C = np.zeros((4, 2))
        C[3, 0] = 1.0
        C[0, 1] = 1.0
        C[1, 0] = 2.0
        out = m.coupled_jacobian(J, 4, [3, 0], [(1, 3, 2.0, 0.0)])
        assert np.allclose(out, J @ C)

    def test_without_mimics_it_selects_columns(self):
        J = RNG.normal(size=(3, 5))
        assert np.allclose(m.coupled_jacobian(J, 5, [4, 2], []), J[:, [4, 2]])

    def test_a_negative_multiplier(self):
        J = RNG.normal(size=(3, 3))
        C = np.array([[1.0], [-1.5], [0.0]])
        assert np.allclose(m.coupled_jacobian(J, 3, [0], [(1, 0, -1.5, 0.0)]), J @ C)


class TestRetargetCost:
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


class TestFingertipIk:
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


class TestFrictionCone:
    def test_inside_the_cone_is_returned_unchanged(self):
        force = np.array([0.1, 0.0, 1.0])
        out = m.project_to_friction_cone(force, [0.0, 0.0, 1.0], 0.5)
        assert out.shape == (3,)
        assert np.allclose(out, force)

    def test_the_projection_lands_on_the_cone(self):
        out = m.project_to_friction_cone([2.0, 0.0, 1.0], [0.0, 0.0, 1.0], 0.5)
        assert np.isclose(np.linalg.norm(out[:2]), 0.5 * out[2])

    def test_pointing_into_the_surface_gives_zero(self):
        out = m.project_to_friction_cone([0.1, 0.0, -1.0], [0.0, 0.0, 1.0], 0.5)
        assert np.allclose(out, 0.0)

    def test_the_projection_is_not_a_rescale(self):
        # Shrinking the whole vector until it fits also shrinks the normal
        # component. The projection increases it.
        force = np.array([2.0, 0.0, 1.0])
        out = m.project_to_friction_cone(force, [0.0, 0.0, 1.0], 0.5)
        assert out[2] > force[2]

    def test_a_tilted_normal_is_handled(self):
        normal = np.array([1.0, 1.0, 1.0])
        unit = normal / np.linalg.norm(normal)
        force = np.array([1.0, -2.0, 0.5])
        out = m.project_to_friction_cone(force, normal, 0.4)
        normal_part = out @ unit
        tangential = np.linalg.norm(out - normal_part * unit)
        assert np.isclose(tangential, 0.4 * normal_part)

    def test_the_normal_need_not_be_unit_length(self):
        short = m.project_to_friction_cone([2.0, 0.0, 1.0], [0.0, 0.0, 1.0], 0.5)
        long = m.project_to_friction_cone([2.0, 0.0, 1.0], [0.0, 0.0, 3.0], 0.5)
        assert np.allclose(short, long)

    def test_frictionless_keeps_only_the_normal_part(self):
        out = m.project_to_friction_cone([2.0, -1.0, 1.0], [0.0, 0.0, 1.0], 0.0)
        assert np.allclose(out, [0.0, 0.0, 1.0])


class TestForceClosure:
    def test_normals_along_the_line_hold_without_friction(self):
        assert m.is_force_closure([[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [-1.0, 0.0]], 0.0)

    def test_a_tilt_inside_the_cone_holds(self):
        tilt = np.deg2rad(30.0)
        n0 = [np.cos(tilt), np.sin(tilt)]
        n1 = [-np.cos(tilt), np.sin(tilt)]
        assert m.is_force_closure([[0.0, 0.0], [1.0, 0.0]], [n0, n1], np.tan(np.deg2rad(40.0)))

    def test_a_tilt_outside_the_cone_slips(self):
        tilt = np.deg2rad(30.0)
        n0 = [np.cos(tilt), np.sin(tilt)]
        n1 = [-np.cos(tilt), np.sin(tilt)]
        assert not m.is_force_closure([[0.0, 0.0], [1.0, 0.0]], [n0, n1], np.tan(np.deg2rad(20.0)))

    def test_opposing_normals_are_not_enough(self):
        # Antipodal, the normals point straight at each other's line, and the
        # grasp still slips: neither cone contains the line joining them.
        assert not m.is_force_closure([[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, -1.0]], 0.5)

    def test_friction_can_rescue_a_tilted_pair(self):
        tilt = np.deg2rad(35.0)
        n0 = [np.cos(tilt), np.sin(tilt)]
        n1 = [-np.cos(tilt), np.sin(tilt)]
        contacts = [[0.0, 0.0], [1.0, 0.0]]
        assert not m.is_force_closure(contacts, n0 and [n0, n1], 0.1)
        assert m.is_force_closure(contacts, [n0, n1], 1.0)

    def test_a_diagonal_pair(self):
        contacts = [[0.0, 0.0], [1.0, 1.0]]
        normals = [[1.0, 1.0], [-1.0, -1.0]]
        assert m.is_force_closure(contacts, normals, 0.0)

    def test_the_normals_need_not_be_unit_length(self):
        assert m.is_force_closure([[0.0, 0.0], [2.0, 0.0]], [[5.0, 0.0], [-0.2, 0.0]], 0.0)

    def test_returns_a_python_bool(self):
        out = m.is_force_closure([[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [-1.0, 0.0]], 0.3)
        assert type(out) is bool
