import numpy as np

from helpers import euler_matrix, is_rotation, transform
from loader import load_track

m = load_track("08_perception")
RNG = np.random.default_rng(8)  # rebound before every test, see conftest.py

K = np.array([[500.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])


def pinhole(p_cam):
    """Reference projection of camera-frame points, for the tests only."""
    p_cam = np.asarray(p_cam, dtype=np.float64)
    return np.stack(
        [K[0, 0] * p_cam[:, 0] / p_cam[:, 2] + K[0, 2], K[1, 1] * p_cam[:, 1] / p_cam[:, 2] + K[1, 2]], axis=1
    )


class TestProjectPoints:
    def test_optical_axis_lands_on_the_principal_point(self):
        pixels, depth, valid = m.project_points([[0.0, 0.0, 2.0]], np.eye(4), K)
        assert pixels.shape == (1, 2)
        assert np.allclose(pixels[0], [320.0, 240.0])
        assert np.isclose(depth[0], 2.0)
        assert valid[0]

    def test_pinhole_with_identity_pose(self):
        p = np.column_stack([RNG.normal(size=(6, 2)), RNG.uniform(1.0, 5.0, size=6)])
        pixels, depth, valid = m.project_points(p, np.eye(4), K)
        assert np.allclose(pixels, pinhole(p))
        assert np.allclose(depth, p[:, 2])
        assert valid.all()

    def test_the_pose_is_inverted_not_applied(self):
        R = euler_matrix(0.4, -0.3, 0.9)
        t = np.array([0.5, -0.2, 1.0])
        T = transform(R=R, p=t)
        p_world = RNG.normal(size=(5, 3)) + [0.0, 0.0, 6.0]
        p_cam = (p_world - t) @ R
        pixels, depth, valid = m.project_points(p_world, T, K)
        assert np.allclose(pixels, pinhole(p_cam))
        assert np.allclose(depth, p_cam[:, 2])

    def test_a_point_behind_the_camera_is_invalid(self):
        points = [[0.0, 0.0, 2.0], [0.1, 0.1, -1.0], [0.1, 0.1, 0.0]]
        pixels, depth, valid = m.project_points(points, np.eye(4), K)
        assert valid.tolist() == [True, False, False]
        assert np.isnan(pixels[1]).all()
        assert np.isnan(pixels[2]).all()
        assert np.isfinite(pixels[0]).all()

    def test_depth_is_z_not_distance(self):
        p = np.array([[3.0, 4.0, 1.0]])
        _, depth, _ = m.project_points(p, np.eye(4), K)
        assert np.isclose(depth[0], 1.0)


class TestBackprojectDepth:
    def test_principal_pixel_lies_on_the_axis(self):
        depth = np.full((480, 640), 2.0)
        points, valid = m.backproject_depth(depth, K)
        assert points.shape == (480, 640, 3)
        assert valid.shape == (480, 640)
        assert np.allclose(points[240, 320], [0.0, 0.0, 2.0])

    def test_round_trip_through_the_pinhole(self):
        depth = RNG.uniform(0.5, 3.0, size=(12, 16))
        points, valid = m.backproject_depth(depth, K)
        assert valid.all()
        flat = points.reshape(-1, 3)
        pixels = pinhole(flat)
        v, u = np.mgrid[0:12, 0:16]
        assert np.allclose(pixels[:, 0], u.ravel())
        assert np.allclose(pixels[:, 1], v.ravel())
        assert np.allclose(flat[:, 2], depth.ravel())

    def test_u_is_the_column(self):
        depth = np.ones((4, 6))
        points, _ = m.backproject_depth(depth, K)
        # moving one column to the right changes x by 1/fx, not y
        assert np.isclose(points[0, 1, 0] - points[0, 0, 0], 1.0 / 500.0)
        assert np.isclose(points[0, 1, 1], points[0, 0, 1])
        assert np.isclose(points[1, 0, 1] - points[0, 0, 1], 1.0 / 400.0)

    def test_missing_depth_gives_no_point(self):
        depth = np.array([[1.0, 0.0], [np.nan, 2.0]])
        points, valid = m.backproject_depth(depth, K)
        assert valid.tolist() == [[True, False], [False, True]]
        assert np.isnan(points[0, 1]).all()
        assert np.isnan(points[1, 0]).all()
        assert np.isfinite(points[0, 0]).all()
        assert np.isfinite(points[1, 1]).all()

    def test_x_scales_with_depth(self):
        near, _ = m.backproject_depth(np.full((3, 3), 1.0), K)
        far, _ = m.backproject_depth(np.full((3, 3), 3.0), K)
        assert np.allclose(far[:, :, :2], 3.0 * near[:, :, :2])


class TestDistortion:
    def test_no_coefficients_is_the_identity(self):
        xy = RNG.normal(size=(7, 2)) * 0.3
        assert np.allclose(m.distort_points(xy, (0.0, 0.0, 0.0, 0.0)), xy)
        assert np.allclose(m.undistort_points(xy, (0.0, 0.0, 0.0, 0.0)), xy)

    def test_radial_moves_along_the_ray(self):
        xy = np.array([[0.3, 0.4]])
        out = m.distort_points(xy, (-0.2, 0.05, 0.0, 0.0))
        r2 = 0.25
        assert np.allclose(out, xy * (1.0 - 0.2 * r2 + 0.05 * r2**2))

    def test_tangential_terms(self):
        out = m.distort_points([[0.3, 0.4]], (0.0, 0.0, 0.01, -0.02))
        r2, x, y = 0.25, 0.3, 0.4
        assert np.isclose(out[0, 0], x + 2 * 0.01 * x * y - 0.02 * (r2 + 2 * x * x))
        assert np.isclose(out[0, 1], y + 0.01 * (r2 + 2 * y * y) + 2 * (-0.02) * x * y)

    def test_the_centre_does_not_move(self):
        assert np.allclose(m.distort_points([[0.0, 0.0]], (0.3, -0.1, 0.02, 0.01)), 0.0)

    def test_undistort_inverts_distort(self):
        coeffs = (-0.25, 0.08, 0.004, -0.003)
        xy = RNG.uniform(-0.6, 0.6, size=(20, 2))
        back = m.undistort_points(m.distort_points(xy, coeffs), coeffs)
        assert np.allclose(back, xy, atol=1e-8)

    def test_one_round_is_not_enough(self):
        coeffs = (-0.25, 0.08, 0.0, 0.0)
        xy = np.array([[0.6, 0.5]])
        xy_d = m.distort_points(xy, coeffs)
        one = m.undistort_points(xy_d, coeffs, iters=1)
        ten = m.undistort_points(xy_d, coeffs, iters=10)
        assert np.linalg.norm(ten - xy) < np.linalg.norm(one - xy)
        assert np.allclose(ten, xy, atol=1e-6)


class TestVoxelDownsample:
    def test_points_in_one_voxel_become_their_centroid(self):
        points = np.array([[0.1, 0.1, 0.1], [0.3, 0.2, 0.4], [0.2, 0.6, 0.1]])
        out = m.voxel_downsample(points, 1.0)
        assert out.shape == (1, 3)
        assert np.allclose(out[0], points.mean(axis=0))

    def test_one_point_per_occupied_voxel(self):
        points = RNG.uniform(0.0, 3.0, size=(200, 3))
        out = m.voxel_downsample(points, 1.0)
        occupied = {tuple(v) for v in np.floor(points).astype(int)}
        assert out.shape == (len(occupied), 3)
        assert {tuple(v) for v in np.floor(out).astype(int)} == occupied

    def test_negative_coordinates_use_floor(self):
        out = m.voxel_downsample([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]], 1.0)
        assert out.shape == (2, 3)

    def test_output_is_sorted_by_voxel_index(self):
        points = RNG.uniform(-2.0, 2.0, size=(50, 3))
        out = m.voxel_downsample(points, 0.5)
        index = np.floor(out / 0.5).astype(int)
        for a, b in zip(index[:-1], index[1:], strict=True):
            assert tuple(a) < tuple(b)

    def test_input_order_does_not_matter(self):
        points = RNG.uniform(-1.0, 1.0, size=(40, 3))
        a = m.voxel_downsample(points, 0.4)
        b = m.voxel_downsample(points[::-1], 0.4)
        assert np.allclose(a, b)

    def test_not_the_voxel_centre(self):
        out = m.voxel_downsample([[0.9, 0.9, 0.9], [0.8, 0.9, 0.9]], 1.0)
        assert np.allclose(out[0], [0.85, 0.9, 0.9])


class TestFitPlane:
    def test_exact_plane(self):
        n_true = np.array([1.0, 2.0, -2.0]) / 3.0
        d_true = 1.5
        # points with dot(n, p) + d = 0
        a = np.array([2.0, -1.0, 0.0]) / 3.0
        b = np.cross(n_true, a)
        uv = RNG.normal(size=(30, 2))
        points = -d_true * n_true + uv[:, :1] * a + uv[:, 1:] * b
        n, d = m.fit_plane(points)
        assert isinstance(d, float)
        assert np.isclose(np.linalg.norm(n), 1.0)
        assert np.allclose(n, n_true)
        assert np.isclose(d, d_true)

    def test_noisy_plane_has_small_residual(self):
        points = np.column_stack([RNG.uniform(-1, 1, 100), RNG.uniform(-1, 1, 100), np.full(100, 0.7)])
        points += RNG.normal(scale=1e-3, size=points.shape)
        n, d = m.fit_plane(points)
        residual = points @ n + d
        assert np.abs(residual).max() < 1e-2
        assert abs(abs(n[2]) - 1.0) < 1e-3

    def test_sign_convention(self):
        points = np.column_stack([RNG.uniform(-1, 1, 20), RNG.uniform(-1, 1, 20), np.full(20, -2.0)])
        n, d = m.fit_plane(points)
        assert d >= 0.0
        assert np.allclose(n, [0.0, 0.0, 1.0])
        assert np.isclose(d, 2.0)

    def test_thinnest_direction_not_longest(self):
        points = np.column_stack([RNG.uniform(-5, 5, 50), RNG.uniform(-0.5, 0.5, 50), np.zeros(50) + 1.0])
        n, _ = m.fit_plane(points)
        assert np.allclose(np.abs(n), [0.0, 0.0, 1.0], atol=1e-9)

    def test_plane_far_from_the_origin(self):
        points = np.column_stack([RNG.uniform(-1, 1, 30), RNG.uniform(-1, 1, 30), np.zeros(30)])
        points += [10.0, 10.0, 10.0]
        n, d = m.fit_plane(points)
        assert np.allclose(np.abs(n), [0.0, 0.0, 1.0])
        assert np.isclose(d, 10.0)


class TestIcpStep:
    def cloud(self, n=40):
        return RNG.uniform(-1.0, 1.0, size=(n, 3))

    def test_aligned_clouds_stay_put(self):
        source = self.cloud()
        T_new, rms = m.icp_step(source, source, np.eye(4))
        assert T_new.shape == (4, 4)
        assert np.allclose(T_new, np.eye(4), atol=1e-9)
        assert isinstance(rms, float)
        assert rms < 1e-9

    def test_exact_correspondences_are_solved_in_one_step(self):
        source = self.cloud()
        R = euler_matrix(0.05, -0.03, 0.04)
        t = np.array([0.02, -0.01, 0.03])
        target = source @ R.T + t
        T_new, rms = m.icp_step(source, target, np.eye(4))
        assert np.allclose(T_new[:3, :3], R, atol=1e-9)
        assert np.allclose(T_new[:3, 3], t, atol=1e-9)
        assert rms < 1e-9

    def test_the_result_is_rigid(self):
        source = self.cloud()
        target = self.cloud(55)
        T_new, _ = m.icp_step(source, target, np.eye(4))
        assert is_rotation(T_new[:3, :3])
        assert np.allclose(T_new[3], [0.0, 0.0, 0.0, 1.0])

    def test_rms_is_measured_after_the_update(self):
        source = self.cloud()
        R = euler_matrix(0.05, -0.03, 0.04)
        target = source @ R.T + [0.1, 0.0, 0.0]
        _, rms = m.icp_step(source, target, np.eye(4))
        assert rms < 1e-9

    def test_the_correction_goes_on_the_left(self):
        source = self.cloud()
        R = euler_matrix(0.05, -0.03, 0.04)
        t = np.array([0.02, -0.01, 0.03])
        target = source @ R.T + t
        # start from a guess that is already rotated: the step must still land on (R, t)
        T0 = transform(R=euler_matrix(0.0, 0.0, 0.03))
        T_new, _ = m.icp_step(source, target, T0)
        assert np.allclose(T_new[:3, :3], R, atol=1e-9)
        assert np.allclose(T_new[:3, 3], t, atol=1e-9)

    def test_each_source_point_picks_its_nearest_target(self):
        source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        target = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
        T_new, rms = m.icp_step(source, target, np.eye(4))
        # the far target point has no partner and must not pull the estimate
        assert np.allclose(T_new, np.eye(4), atol=1e-9)
        assert rms < 1e-9

    def test_a_step_reduces_the_error(self):
        source = self.cloud(80)
        R = euler_matrix(0.2, -0.1, 0.15)
        target = source @ R.T + [0.1, 0.05, -0.1]
        moved = source
        d2 = ((moved[:, None, :] - target[None, :, :]) ** 2).sum(axis=2)
        before = np.sqrt(np.mean(d2.min(axis=1)))
        _, rms = m.icp_step(source, target, np.eye(4))
        assert rms < before
