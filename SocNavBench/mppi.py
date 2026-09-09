import numpy as np
import time
from scipy.ndimage import distance_transform_edt
from trajectory_predictors import make_trajectory_predictor

try:
    import torch
except ImportError:
    torch = None

def _torch_no_grad():
    if torch is None:
        def decorator(function):
            return function
        return decorator
    return torch.no_grad()


class OccupancyGridDistanceField:
    """Metric clearance lookup for SocNavBench traversibility grids."""

    def __init__(self, traversible, resolution, robot_radius):
        self.traversible = np.asarray(traversible, dtype=bool)
        if self.traversible.ndim != 2:
            raise ValueError("traversible must be a 2-D grid")
        self.resolution = float(resolution)
        self.robot_radius = float(robot_radius)
        self.distance_field = (
            distance_transform_edt(self.traversible).astype(np.float32)
            * self.resolution
        )

    def clearance_at(self, position):
        position = np.asarray(position, dtype=float)
        col = int(np.floor(position[0] / self.resolution))
        row = int(np.floor(position[1] / self.resolution))
        if (
            row < 0
            or col < 0
            or row >= self.distance_field.shape[0]
            or col >= self.distance_field.shape[1]
        ):
            return -self.robot_radius
        return float(self.distance_field[row, col]) - self.robot_radius


class MPPIController:
    def __init__(
        self,
        horizon=10,
        num_iterations=10,
        num_samples=100,
        temp=0.95,
        std_v=1.0,
        std_w=1.0,
        dt=0.1,
        dynamic_history_len=20,
        goal_radius=0.3,
        goal_weight=10.0,
        terminal_goal_weight=100.0,
        goal_bonus=100.0,
        progress_weight=5.0,
        heading_weight=2.0,
        desired_speed=1.0,
        goal_speed_gain=1.0,
        slowdown_radius=1.0,
        speed_weight=2.0,
        stop_speed_weight=5.0,
        control_effort_weight=1.0,
        control_smoothing_weight=0.0,
        action_noise_beta=1.0,
        initial_sequence='zeros',
        initial_sequence_speed_fraction=0.5,
        initial_sequence_angular_fraction=0.5,
        static_obs_clearance=0.3,
        static_obs_weight=100.0,
        dynamic_obs_clearance=0.3,
        dynamic_obs_weight=100.0,
        v_min=0.0,
        v_max=1.5,
        w_min=-2.84,
        w_max=2.84,
        dynamic_prediction_method='orca',
        orca_neighbor_dist=10.0,
        orca_max_neighbors=10,
        orca_time_horizon=5.0,
        orca_time_horizon_obst=5.0,
    ):
        self.horizon = horizon
        self.num_iterations = num_iterations
        self.num_samples = num_samples
        self.temp = temp
        self.std_diag = np.array([std_v**2, std_w**2])
        self.std = np.diag(self.std_diag)
        self.dt = dt
        # self.v_limits = (-0.22, 0.22)
        # self.w_limits = (-2.84, 2.84)
        self.v_limits = (v_min, v_max) # turtlebot is (0.0, 0.22)
        self.w_limits = (w_min, w_max)
        self.dynamic_history_len = dynamic_history_len
        self.cached_control_seq = np.zeros((self.horizon, 2))
        self.goal_radius = goal_radius
        self.goal_weight = goal_weight
        self.terminal_goal_weight = terminal_goal_weight
        self.goal_bonus = goal_bonus
        self.progress_weight = progress_weight
        self.heading_weight = heading_weight
        self.desired_speed = desired_speed
        self.goal_speed_gain = goal_speed_gain
        self.slowdown_radius = slowdown_radius
        self.speed_weight = speed_weight
        self.stop_speed_weight = stop_speed_weight
        self.control_effort_weight = control_effort_weight
        self.control_smoothing_weight = control_smoothing_weight
        self.action_noise_beta = action_noise_beta
        self.initial_sequence = initial_sequence
        self.initial_sequence_speed_fraction = initial_sequence_speed_fraction
        self.initial_sequence_angular_fraction = initial_sequence_angular_fraction
        self.static_obs_clearance = static_obs_clearance
        self.static_obs_weight = static_obs_weight
        self.dynamic_obs_clearance = dynamic_obs_clearance
        self.dynamic_obs_weight = dynamic_obs_weight
        self.dynamic_prediction_method = dynamic_prediction_method
        self.orca_neighbor_dist = orca_neighbor_dist
        self.orca_max_neighbors = orca_max_neighbors
        self.orca_time_horizon = orca_time_horizon
        self.orca_time_horizon_obst = orca_time_horizon_obst
        self.trajectory_predictor = make_trajectory_predictor(
            dynamic_prediction_method,
            horizon,
            dt,
            neighbor_dist=orca_neighbor_dist,
            max_neighbors=orca_max_neighbors,
            time_horizon=orca_time_horizon,
            time_horizon_obst=orca_time_horizon_obst,
            max_speed=v_max,
        )

    def reset_cached_control_sequence(self, self_state=None):
        if self.initial_sequence == 'goal_directed' and self_state is not None:
            self.cached_control_seq = self._build_goal_directed_initial_sequence(self_state)
        else:
            self.cached_control_seq = np.zeros((self.horizon, 2))

    def _build_goal_directed_initial_sequence(self, self_state):
        controls = np.zeros((self.horizon, 2))
        dx = self_state.gx - self_state.px
        dy = self_state.gy - self_state.py
        heading_error = np.arctan2(np.sin(np.arctan2(dy, dx) - self_state.theta),
                                   np.cos(np.arctan2(dy, dx) - self_state.theta))
        linear_speed = np.clip(
            self.v_limits[1] * self.initial_sequence_speed_fraction,
            *self.v_limits,
        )
        angular_speed = max(abs(self.w_limits[0]), abs(self.w_limits[1])) * self.initial_sequence_angular_fraction
        if angular_speed > 0:
            rotate_steps = int(np.ceil(abs(heading_error) / (angular_speed * self.dt)))
            rotate_steps = min(rotate_steps, self.horizon)
            controls[:rotate_steps, 1] = np.sign(heading_error) * angular_speed
            controls[rotate_steps:, 0] = linear_speed
        else:
            controls[:, 0] = linear_speed
        return controls

    def next_state_dynamics(self, x_t, u_t, dt):
        x_t_1 = x_t.copy()
        # Match SocNavBench's Dubins3D forward-Euler integration exactly.
        x_t_1[0] = x_t[0] + u_t[0] * np.cos(x_t[2]) * dt
        x_t_1[1] = x_t[1] + u_t[0] * np.sin(x_t[2]) * dt
        x_t_1[2] = x_t[2] + u_t[1] * dt
        return x_t_1

    def loss(
        self,
        x_t,
        goal,
        u_t,
        obstacles,
        dynamic_obstacles,
        dynamic_obstacle_predictions,
        dynamic_obstacle_histories,
        dynamic_obstacle_history_mask,
        t,
        x_prev=None,
        u_prev=None,
        static_obstacle_map=None,
    ):
        distance_to_goal = np.linalg.norm(x_t[0:2] - goal)
        goal_error = max(0.0, distance_to_goal - self.goal_radius)
        goal_cost = self.goal_weight * goal_error

        if obstacles.shape[0] > 0:
            center_dists = np.linalg.norm(x_t[0:2] - obstacles[:, 0:2], axis=1)
            edge_dists = center_dists - obstacles[:, 2]
            in_range = edge_dists < self.static_obs_clearance
            dist_costs = np.where(
                edge_dists < 0.0,
                1e4,
                np.where(in_range, self.static_obs_weight * np.exp(-edge_dists / 0.8), 0.0),
            )
            dist_cost = np.sum(dist_costs)
        else:
            dist_cost = 0.0
        if static_obstacle_map is not None:
            static_clearance = static_obstacle_map.clearance_at(x_t[0:2])
            if static_clearance < 0.0:
                dist_cost += 1e4
            elif static_clearance < self.static_obs_clearance:
                dist_cost += self.static_obs_weight * np.exp(
                    -static_clearance / 0.8
                )

        if dynamic_obstacles.shape[0] > 0:
            # dynamic_obstacle_histories has shape (num_dynamic_obstacles, history_len, 2).
            # dynamic_obstacle_history_mask marks which padded history slots are valid.
            projected_dynamic_obstacle_pose = self.dynamic_obstacle_poses_at_timestep(
                dynamic_obstacles,
                dynamic_obstacle_predictions,
                t,
            )
            dynamic_center_dists = np.linalg.norm(x_t[0:2] - projected_dynamic_obstacle_pose, axis=1)
            dynamic_edge_dists = dynamic_center_dists - dynamic_obstacles[:, 2]
            dynamic_in_range = dynamic_edge_dists < self.dynamic_obs_clearance
            dynamic_costs = np.where(
                dynamic_edge_dists < 0.0,
                1e4,
                np.where(dynamic_in_range, self.dynamic_obs_weight * np.exp(-dynamic_edge_dists / 0.8), 0.0),
            )
            dynamic_dist_cost = np.sum(dynamic_costs) #* (0.95**t)
        else:
            dynamic_dist_cost = 0.0

        if distance_to_goal < self.slowdown_radius:
            speed_cost = self.stop_speed_weight * u_t[0] ** 2
        else:
            v_des = min(self.desired_speed, self.goal_speed_gain * goal_error)
            speed_cost = self.speed_weight * (v_des - u_t[0]) ** 2

        angle_to_goal = np.arctan2(goal[1] - x_t[1], goal[0] - x_t[0])
        heading_error = np.arctan2(np.sin(angle_to_goal - x_t[2]), np.cos(angle_to_goal - x_t[2]))
        heading_cost = self.heading_weight * min(goal_error, 1.0) * heading_error ** 2

        if x_prev is not None:
            previous_goal_error = max(0.0, np.linalg.norm(x_prev[0:2] - goal) - self.goal_radius)
            progress_cost = self.progress_weight * (goal_error - previous_goal_error)
        else:
            progress_cost = 0.0

        control_effort_cost = 0.5 * self.control_effort_weight * self.temp * (
            u_t[0] ** 2 * self.std_diag[0] + u_t[1] ** 2 * self.std_diag[1]
        )
        if u_prev is not None:
            control_smoothing_cost = self.control_smoothing_weight * np.sum((u_t - u_prev) ** 2)
        else:
            control_smoothing_cost = 0.0
        return (
            goal_cost + dist_cost + dynamic_dist_cost + control_effort_cost + speed_cost + heading_cost +
            progress_cost + control_smoothing_cost
        )

    def terminal_cost(self, x_t, goal):
        distance_to_goal = np.linalg.norm(x_t[0:2] - goal)
        goal_error = max(0.0, distance_to_goal - self.goal_radius)
        if goal_error == 0.0:
            return -self.goal_bonus
        return self.terminal_goal_weight * goal_error

    def control(self, state, obstacles=None):
        control_seq = np.concatenate((self.cached_control_seq[1:], self.cached_control_seq[-1:]), axis=0)
        x_0 = self._robot_pose_from_state(state)
        goal = self._goal_from_state(state)
        obstacles = self._obstacles_to_array(obstacles)
        dynamic_obstacles_arr, dynamic_obstacle_histories, dynamic_obstacle_history_mask = (
            self._dynamic_obstacles_to_arrays(state)
        )
        dynamic_obstacle_predictions = self.predict_dynamic_obstacle_poses(state, dynamic_obstacles_arr)
        static_obstacle_map = getattr(state, "static_obstacle_map", None)

        for i in range(self.num_iterations):
            J_sample = np.zeros(self.num_samples)
            eps_sample = np.random.multivariate_normal(np.zeros(2), self.std, size=(self.num_samples, self.horizon))
            controls_sample = self._build_noisy_control_samples(control_seq, eps_sample)
            eps_sample = controls_sample - control_seq[np.newaxis, :, :]
            for m in range(self.num_samples):
                J_m = 0.0
                curr_state = x_0.copy()
                reached_goal = False
                for t in range(self.horizon):
                    v_t = controls_sample[m, t]
                    u_prev = controls_sample[m, t - 1] if t > 0 else np.zeros(2)
                    prev_state = curr_state.copy()
                    curr_state = self.next_state_dynamics(curr_state, v_t, self.dt)
                    if not reached_goal:
                        J_m += self.loss(
                            curr_state,
                            goal,
                            v_t,
                            obstacles,
                            dynamic_obstacles_arr,
                            dynamic_obstacle_predictions,
                            dynamic_obstacle_histories,
                            dynamic_obstacle_history_mask,
                            t,
                            prev_state,
                            u_prev,
                            static_obstacle_map,
                        )
                        if np.linalg.norm(curr_state[0:2] - goal) <= self.goal_radius:
                            J_m -= self.goal_bonus
                            reached_goal = True
                if not reached_goal:
                    J_m += self.terminal_cost(curr_state, goal)
                J_sample[m] = J_m

            rho = np.min(J_sample)
            denom = np.sum(np.exp(-(J_sample - rho) / self.temp))
            omegas = np.exp(-(J_sample - rho) / self.temp) / denom
            for t in range(self.horizon):
                control_seq[t] += np.sum(omegas[:, None]*eps_sample[:, t], axis=0)
            control_seq[:, 0] = np.clip(control_seq[:, 0], *self.v_limits)
            control_seq[:, 1] = np.clip(control_seq[:, 1], *self.w_limits)
            self.cached_control_seq = control_seq
        return control_seq

    def control_action(self, state, obstacles=None):
        return self.control(state, obstacles)[0].copy()

    def _build_noisy_control_samples(self, control_seq, eps_sample):
        controls_sample = np.empty_like(eps_sample)
        beta = np.clip(self.action_noise_beta, 0.0, 1.0)
        controls_sample[:, 0] = control_seq[0] + eps_sample[:, 0]
        controls_sample[:, 0, 0] = np.clip(controls_sample[:, 0, 0], *self.v_limits)
        controls_sample[:, 0, 1] = np.clip(controls_sample[:, 0, 1], *self.w_limits)
        for t in range(1, self.horizon):
            noisy_control = control_seq[t] + eps_sample[:, t]
            controls_sample[:, t] = beta * noisy_control + (1.0 - beta) * controls_sample[:, t - 1]
            controls_sample[:, t, 0] = np.clip(controls_sample[:, t, 0], *self.v_limits)
            controls_sample[:, t, 1] = np.clip(controls_sample[:, t, 1], *self.w_limits)
        return controls_sample

    @staticmethod
    def _obstacles_to_array(obstacles):
        if hasattr(obstacles, "occupancy_grid"):
            return np.empty((0, 3), dtype=float)
        if obstacles is None or len(obstacles) == 0:
            return np.empty((0, 3), dtype=float)
        return np.asarray(obstacles, dtype=float)

    @staticmethod
    def _robot_pose_from_state(state):
        self_state = state.self_state
        return np.array((self_state.px, self_state.py, self_state.theta), dtype=float)

    @staticmethod
    def _goal_from_state(state):
        self_state = state.self_state
        return np.array((self_state.gx, self_state.gy), dtype=float)

    def predict_dynamic_obstacle_poses(self, state, dynamic_obstacles):
        return self.trajectory_predictor.predict(state, dynamic_obstacles)

    @staticmethod
    def dynamic_obstacle_poses_at_timestep(dynamic_obstacles, dynamic_obstacle_predictions, t):
        if dynamic_obstacle_predictions.shape[0] == 0:
            return dynamic_obstacles[:, 0:2]
        return dynamic_obstacle_predictions[min(t, dynamic_obstacle_predictions.shape[0] - 1)]

    def _dynamic_obstacles_to_arrays(self, dynamic_obstacles):
        human_states = dynamic_obstacles.human_states
        if len(human_states) == 0:
            return (
                np.empty((0, 5), dtype=float),
                np.empty((0, self.dynamic_history_len, 2), dtype=float),
                np.empty((0, self.dynamic_history_len), dtype=bool),
            )

        robot_radius = dynamic_obstacles.self_state.radius
        dynamic_obstacles_arr = np.array(
            [
                (
                    human_state.position[0],
                    human_state.position[1],
                    human_state.radius + robot_radius,
                    human_state.velocity[0],
                    human_state.velocity[1],
                )
                for human_state in human_states
            ],
            dtype=float,
        )
        histories = np.zeros((len(human_states), self.dynamic_history_len, 2), dtype=float)
        history_mask = np.zeros((len(human_states), self.dynamic_history_len), dtype=bool)

        return dynamic_obstacles_arr, histories, history_mask


class GPUMPPIController:
    def __init__(
        self,
        horizon=10,
        num_iterations=10,
        num_samples=100,
        temp=0.95,
        std_v=1.0,
        std_w=1.0,
        dt=0.1,
        device=None,
        compile_rollout=True,
        warmup_obstacles=None,
        warmup_dynamic_obstacles=None,
        dynamic_history_len=20,
        goal_radius=0.3,
        goal_weight=10.0,
        terminal_goal_weight=100.0,
        goal_bonus=100.0,
        progress_weight=5.0,
        heading_weight=2.0,
        desired_speed=1.0,
        goal_speed_gain=1.0,
        slowdown_radius=1.0,
        speed_weight=2.0,
        stop_speed_weight=5.0,
        control_effort_weight=1.0,
        control_smoothing_weight=0.0,
        action_noise_beta=1.0,
        initial_sequence='zeros',
        initial_sequence_speed_fraction=0.5,
        initial_sequence_angular_fraction=0.5,
        static_obs_clearance=0.3,
        static_obs_weight=100.0,
        dynamic_obs_clearance=0.3,
        dynamic_obs_weight=100.0,
        v_min=0.0,
        v_max=1.5,
        w_min=-2.84,
        w_max=2.84,
        dynamic_prediction_method='orca',
        orca_neighbor_dist=10.0,
        orca_max_neighbors=10,
        orca_time_horizon=5.0,
        orca_time_horizon_obst=5.0,
    ):
        if torch is None:
            raise ImportError(
                "PyTorch is required for GPUMPPIController; use "
                "MPPIController for NumPy CPU rollouts."
            )
        self.horizon = horizon
        self.num_iterations = num_iterations
        self.num_samples = num_samples
        self.temp = temp
        self.dt = dt
        self.v_limits = (v_min, v_max) # turtlebot is (0.0, 0.22)
        self.w_limits = (w_min, w_max)
        self._compile_rollout = bool(
            compile_rollout and hasattr(torch, "compile")
        )
        self._rollout_fn = None
        self.dynamic_history_len = dynamic_history_len
        self.goal_radius = goal_radius
        self.goal_weight = goal_weight
        self.terminal_goal_weight = terminal_goal_weight
        self.goal_bonus = goal_bonus
        self.progress_weight = progress_weight
        self.heading_weight = heading_weight
        self.desired_speed = desired_speed
        self.goal_speed_gain = goal_speed_gain
        self.slowdown_radius = slowdown_radius
        self.speed_weight = speed_weight
        self.stop_speed_weight = stop_speed_weight
        self.control_effort_weight = control_effort_weight
        self.control_smoothing_weight = control_smoothing_weight
        self.action_noise_beta = action_noise_beta
        self.initial_sequence = initial_sequence
        self.initial_sequence_speed_fraction = initial_sequence_speed_fraction
        self.initial_sequence_angular_fraction = initial_sequence_angular_fraction
        self.static_obs_clearance = static_obs_clearance
        self.static_obs_weight = static_obs_weight
        self.dynamic_obs_clearance = dynamic_obs_clearance
        self.dynamic_obs_weight = dynamic_obs_weight
        self.dynamic_prediction_method = dynamic_prediction_method
        self.orca_neighbor_dist = orca_neighbor_dist
        self.orca_max_neighbors = orca_max_neighbors
        self.orca_time_horizon = orca_time_horizon
        self.orca_time_horizon_obst = orca_time_horizon_obst
        self.trajectory_predictor = make_trajectory_predictor(
            dynamic_prediction_method,
            horizon,
            dt,
            neighbor_dist=orca_neighbor_dist,
            max_neighbors=orca_max_neighbors,
            time_horizon=orca_time_horizon,
            time_horizon_obst=orca_time_horizon_obst,
            max_speed=v_max,
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.std_diag = torch.tensor([std_v**2, std_w**2], device=self.device, dtype=torch.float32)
        self.cached_control_seq = torch.zeros((horizon, 2), device=self.device, dtype=torch.float32)
        self._controls = torch.empty((num_samples, horizon, 2), device=self.device, dtype=torch.float32)
        self._eps = torch.empty((num_samples, horizon, 2), device=self.device, dtype=torch.float32)
        self._state = torch.empty((num_samples, 3), device=self.device, dtype=torch.float32)
        self._costs = torch.empty(num_samples, device=self.device, dtype=torch.float32)
        self._weights = torch.empty(num_samples, device=self.device, dtype=torch.float32)
        self._control_delta = torch.empty((horizon, 2), device=self.device, dtype=torch.float32)
        self._empty_obstacles = torch.empty((0, 3), device=self.device, dtype=torch.float32)
        self._empty_dynamic = torch.empty((0, 5), device=self.device, dtype=torch.float32)
        self._empty_histories = torch.empty(
            (0, self.dynamic_history_len, 2), device=self.device, dtype=torch.float32
        )
        self._empty_history_mask = torch.empty(
            (0, self.dynamic_history_len), device=self.device, dtype=torch.bool
        )
        self._empty_predictions = torch.empty(
            (horizon, 0, 2), device=self.device, dtype=torch.float32
        )
        self._empty_sdf = torch.empty((0, 0), device=self.device, dtype=torch.float32)
        self._cached_sdf = None
        self._cached_sdf_key = None
        self._cached_sdf_resolution = 1.0
        self._cached_sdf_robot_radius = 0.0
        self._warmup_obstacles = warmup_obstacles
        self._warmup_dynamic_obstacles = warmup_dynamic_obstacles

        if self.device.type == "cuda":
            self._warmup()

    def reset_cached_control_sequence(self, self_state=None):
        if self.initial_sequence == 'goal_directed' and self_state is not None:
            initial = MPPIController._build_goal_directed_initial_sequence(self, self_state)
            self.cached_control_seq.copy_(
                torch.as_tensor(initial, device=self.device, dtype=torch.float32)
            )
        else:
            self.cached_control_seq.zero_()

    def _warmup(self):
        start = time.perf_counter()

        class _DummySelfState:
            px = 0.0
            py = 0.0
            theta = 0.0
            gx = 0.0
            gy = 0.0
            radius = 0.3

        class _DummyState:
            self_state = _DummySelfState()
            human_states = []

        self.control(_DummyState(), self._warmup_obstacles)
        print(f"GPU warmup completed in {time.perf_counter() - start:.2f}s")

    @staticmethod
    def _next_state_batch(state, control, dt):
        v = control[:, 0]
        w = control[:, 1]
        theta = state[:, 2]
        return torch.stack(
            (
                state[:, 0] + v * torch.cos(theta) * dt,
                state[:, 1] + v * torch.sin(theta) * dt,
                theta + w * dt,
            ),
            dim=1,
        )

    @staticmethod
    def _loss_batch(
        state,
        goal,
        control,
        obstacles,
        dynamic_obstacles,
        dynamic_obstacle_predictions,
        dynamic_obstacle_histories,
        dynamic_obstacle_history_mask,
        static_distance_field,
        static_map_resolution,
        robot_radius,
        std_diag,
        temp,
        t,
        previous_state,
        previous_control,
        goal_radius,
        goal_weight,
        progress_weight,
        heading_weight,
        desired_speed,
        goal_speed_gain,
        slowdown_radius,
        speed_weight,
        stop_speed_weight,
        control_effort_weight,
        static_obs_clearance,
        static_obs_weight,
        dynamic_obs_clearance,
        dynamic_obs_weight,
        control_smoothing_weight,
    ):
        distance_to_goal = torch.linalg.norm(state[:, :2] - goal, dim=1)
        goal_error = torch.clamp(distance_to_goal - goal_radius, min=0.0)
        goal_cost = goal_weight * goal_error
    
        if obstacles.shape[0] > 0:
            diff = state[:, None, :2] - obstacles[None, :, :2]
            center_dists = torch.linalg.norm(diff, dim=2)
            edge_dists = center_dists - obstacles[None, :, 2]
            in_range = edge_dists < static_obs_clearance
            collision_cost = torch.full_like(edge_dists, 1e4)
            clearance_cost = static_obs_weight * torch.exp(-edge_dists / 0.8)
            dist_cost = torch.where(
                edge_dists < 0.0,
                collision_cost,
                torch.where(in_range, clearance_cost, torch.zeros_like(edge_dists)),
            ).sum(dim=1)
        else:
            dist_cost = torch.zeros(state.shape[0], device=state.device, dtype=state.dtype)
        if static_distance_field.numel() > 0:
            cols = torch.floor(state[:, 0] / static_map_resolution).long()
            rows = torch.floor(state[:, 1] / static_map_resolution).long()
            inside = (
                (rows >= 0)
                & (cols >= 0)
                & (rows < static_distance_field.shape[0])
                & (cols < static_distance_field.shape[1])
            )
            safe_rows = rows.clamp(0, static_distance_field.shape[0] - 1)
            safe_cols = cols.clamp(0, static_distance_field.shape[1] - 1)
            static_clearance = (
                static_distance_field[safe_rows, safe_cols] - robot_radius
            )
            static_clearance = torch.where(
                inside,
                static_clearance,
                torch.full_like(static_clearance, -robot_radius),
            )
            static_cost = torch.where(
                static_clearance < 0.0,
                torch.full_like(static_clearance, 1e4),
                torch.where(
                    static_clearance < static_obs_clearance,
                    static_obs_weight * torch.exp(-static_clearance / 0.8),
                    torch.zeros_like(static_clearance),
                ),
            )
            dist_cost = dist_cost + static_cost

        if dynamic_obstacles.shape[0] > 0:
            # Match CPU MPPIController.dynamic_obstacle_poses_at_timestep.
            if dynamic_obstacle_predictions.shape[0] == 0:
                projected_dynamic_obstacle_pose = dynamic_obstacles[:, 0:2]
            else:
                timestep = min(t, int(dynamic_obstacle_predictions.shape[0]) - 1)
                projected_dynamic_obstacle_pose = dynamic_obstacle_predictions[timestep]
            dynamic_diff = state[:, None, :2] - projected_dynamic_obstacle_pose[None, :, :]
            dynamic_center_dists = torch.linalg.norm(dynamic_diff, dim=2)
            dynamic_edge_dists = dynamic_center_dists - dynamic_obstacles[None, :, 2]
            dynamic_in_range = dynamic_edge_dists < dynamic_obs_clearance
            dynamic_collision_cost = torch.full_like(dynamic_edge_dists, 1e4)
            dynamic_clearance_cost = dynamic_obs_weight * torch.exp(-dynamic_edge_dists / 0.8)
            dynamic_dist_cost = torch.where(
                dynamic_edge_dists < 0.0,
                dynamic_collision_cost,
                torch.where(dynamic_in_range, dynamic_clearance_cost, torch.zeros_like(dynamic_edge_dists)),
            ).sum(dim=1)
        else:
            dynamic_dist_cost = torch.zeros(state.shape[0], device=state.device, dtype=state.dtype)
        v_des = torch.minimum(
            torch.full_like(goal_error, desired_speed),
            goal_speed_gain * goal_error,
        )
        speed_tracking_cost = speed_weight * (v_des - control[:, 0]) ** 2
        stop_cost = stop_speed_weight * control[:, 0] ** 2
        speed_cost = torch.where(distance_to_goal < slowdown_radius, stop_cost, speed_tracking_cost)

        angle_to_goal = torch.atan2(goal[1] - state[:, 1], goal[0] - state[:, 0])
        heading_error = torch.atan2(torch.sin(angle_to_goal - state[:, 2]), torch.cos(angle_to_goal - state[:, 2]))
        heading_cost = heading_weight * torch.minimum(goal_error, torch.ones_like(goal_error)) * heading_error ** 2

        previous_goal_error = torch.clamp(torch.linalg.norm(previous_state[:, :2] - goal, dim=1) - goal_radius, min=0.0)
        progress_cost = progress_weight * (goal_error - previous_goal_error)

        control_effort = 0.5 * control_effort_weight * temp * (
            control[:, 0] ** 2 * std_diag[0] + control[:, 1] ** 2 * std_diag[1]
        )
        control_smoothing = control_smoothing_weight * torch.sum((control - previous_control) ** 2, dim=1)
        return (
            goal_cost + dist_cost + dynamic_dist_cost + control_effort + speed_cost + heading_cost + progress_cost +
            control_smoothing
        )

    @staticmethod
    def _terminal_cost(state, goal, goal_radius, terminal_goal_weight, goal_bonus):
        distance_to_goal = torch.linalg.norm(state[:, :2] - goal, dim=1)
        goal_error = torch.clamp(distance_to_goal - goal_radius, min=0.0)
        return torch.where(
            goal_error == 0.0,
            torch.full_like(goal_error, -float(goal_bonus)),
            terminal_goal_weight * goal_error,
        )

    def _rollout_update_impl(
        self,
        control_seq,
        x0,
        goal,
        obstacles,
        dynamic_obstacles,
        dynamic_obstacle_predictions,
        dynamic_obstacle_histories,
        dynamic_obstacle_history_mask,
        static_distance_field,
        static_map_resolution,
        robot_radius,
    ):
        noise_scale = torch.sqrt(self.std_diag)
        self._eps.normal_()
        self._eps.mul_(noise_scale)

        self._controls[:, 0].copy_(control_seq[0].unsqueeze(0).add(self._eps[:, 0]))
        self._controls[:, 0, 0].clamp_(*self.v_limits)
        self._controls[:, 0, 1].clamp_(*self.w_limits)
        beta = min(max(self.action_noise_beta, 0.0), 1.0)
        for t in range(1, self.horizon):
            noisy_control = control_seq[t].unsqueeze(0).add(self._eps[:, t])
            self._controls[:, t].copy_(beta * noisy_control + (1.0 - beta) * self._controls[:, t - 1])
            self._controls[:, t, 0].clamp_(*self.v_limits)
            self._controls[:, t, 1].clamp_(*self.w_limits)
        self._eps.copy_(self._controls.sub(control_seq.unsqueeze(0)))

        self._state.copy_(x0.unsqueeze(0).expand(self.num_samples, -1))
        self._costs.zero_()
        active_mask = torch.ones(self.num_samples, device=self.device, dtype=torch.bool)

        for t in range(self.horizon):
            previous_state = self._state.clone()
            if t == 0:
                previous_control = torch.zeros_like(self._controls[:, t])
            else:
                previous_control = self._controls[:, t - 1]
            self._state.copy_(self._next_state_batch(self._state, self._controls[:, t], self.dt))
            step_cost = self._loss_batch(
                self._state,
                goal,
                self._controls[:, t],
                obstacles,
                dynamic_obstacles,
                dynamic_obstacle_predictions,
                dynamic_obstacle_histories,
                dynamic_obstacle_history_mask,
                static_distance_field,
                static_map_resolution,
                robot_radius,
                self.std_diag,
                self.temp,
                t,
                previous_state,
                previous_control,
                self.goal_radius,
                self.goal_weight,
                self.progress_weight,
                self.heading_weight,
                self.desired_speed,
                self.goal_speed_gain,
                self.slowdown_radius,
                self.speed_weight,
                self.stop_speed_weight,
                self.control_effort_weight,
                self.static_obs_clearance,
                self.static_obs_weight,
                self.dynamic_obs_clearance,
                self.dynamic_obs_weight,
                self.control_smoothing_weight,
            )
            self._costs.add_(torch.where(active_mask, step_cost, torch.zeros_like(step_cost)))
            reached_now = active_mask & (torch.linalg.norm(self._state[:, :2] - goal, dim=1) <= self.goal_radius)
            self._costs.add_(torch.where(
                reached_now,
                torch.full_like(self._costs, -self.goal_bonus),
                torch.zeros_like(self._costs),
            ))
            active_mask = active_mask & torch.logical_not(reached_now)
        terminal_cost = self._terminal_cost(
            self._state,
            goal,
            self.goal_radius,
            self.terminal_goal_weight,
            self.goal_bonus,
        )
        self._costs.add_(torch.where(active_mask, terminal_cost, torch.zeros_like(terminal_cost)))

        rho = self._costs.min()
        self._weights.copy_(torch.exp(-(self._costs - rho) / self.temp))
        self._weights.div_(self._weights.sum())
        torch.sum(self._weights[:, None, None] * self._eps, dim=0, out=self._control_delta)
        return control_seq + self._control_delta, self._state

    def _get_rollout_fn(self):
        if self._rollout_fn is not None:
            return self._rollout_fn
        if self._compile_rollout and self.device.type == "cuda":
            self._rollout_fn = torch.compile(self._rollout_update_impl, mode="reduce-overhead")
        else:
            self._rollout_fn = self._rollout_update_impl
        return self._rollout_fn

    @_torch_no_grad()
    def control(self, state, obstacles=None):
        control_seq = torch.cat(
            (self.cached_control_seq[1:], self.cached_control_seq[-1:]),
            dim=0,
        )
        x0 = torch.as_tensor(
            MPPIController._robot_pose_from_state(state),
            device=self.device,
            dtype=torch.float32,
        )
        goal_xy = torch.as_tensor(
            MPPIController._goal_from_state(state),
            device=self.device,
            dtype=torch.float32,
        )

        obstacles_np = MPPIController._obstacles_to_array(obstacles)
        if obstacles_np.shape[0] > 0:
            obstacles_t = torch.as_tensor(obstacles_np, device=self.device, dtype=torch.float32)
        else:
            obstacles_t = self._empty_obstacles

        (
            dynamic_obstacles_t,
            dynamic_obstacle_histories_t,
            dynamic_obstacle_history_mask_t,
            dynamic_obstacles_np,
        ) = self._dynamic_obstacles_to_tensors(state)
        dynamic_obstacle_predictions_t = self._predict_dynamic_obstacle_pose_tensors(
            state, dynamic_obstacles_np
        )
        static_distance_field_t, static_map_resolution, robot_radius = (
            self._static_map_tensors(state)
        )

        rollout = self._get_rollout_fn()
        for _ in range(self.num_iterations):
            control_seq, _ = rollout(
                control_seq,
                x0,
                goal_xy,
                obstacles_t,
                dynamic_obstacles_t,
                dynamic_obstacle_predictions_t,
                dynamic_obstacle_histories_t,
                dynamic_obstacle_history_mask_t,
                static_distance_field_t,
                static_map_resolution,
                robot_radius,
            )
            control_seq[:, 0].clamp_(*self.v_limits)
            control_seq[:, 1].clamp_(*self.w_limits)

        self.cached_control_seq.copy_(control_seq)
        return self.cached_control_seq.cpu().numpy()

    def control_action(self, state, obstacles=None):
        return self.control(state, obstacles)[0].copy()

    def _static_map_tensors(self, state):
        static_obstacle_map = getattr(state, "static_obstacle_map", None)
        if static_obstacle_map is None:
            return (
                self._empty_sdf,
                1.0,
                float(state.self_state.radius),
            )
        key = id(static_obstacle_map)
        if self._cached_sdf_key != key or self._cached_sdf is None:
            self._cached_sdf = torch.as_tensor(
                static_obstacle_map.distance_field,
                device=self.device,
                dtype=torch.float32,
            )
            self._cached_sdf_key = key
            self._cached_sdf_resolution = float(static_obstacle_map.resolution)
            self._cached_sdf_robot_radius = float(static_obstacle_map.robot_radius)
        return (
            self._cached_sdf,
            self._cached_sdf_resolution,
            self._cached_sdf_robot_radius,
        )

    def _predict_dynamic_obstacle_pose_tensors(self, state, dynamic_obstacles_np):
        predictions = self.trajectory_predictor.predict(state, dynamic_obstacles_np)
        if predictions.shape[0] == 0:
            return self._empty_predictions
        return torch.as_tensor(
            predictions,
            device=self.device,
            dtype=torch.float32,
        )

    def _dynamic_obstacles_to_tensors(self, state):
        dynamic_obstacles_np, histories_np, history_mask_np = (
            MPPIController._dynamic_obstacles_to_arrays(self, state)
        )
        if dynamic_obstacles_np.shape[0] == 0:
            return (
                self._empty_dynamic,
                self._empty_histories,
                self._empty_history_mask,
                dynamic_obstacles_np,
            )
        return (
            torch.as_tensor(dynamic_obstacles_np, device=self.device, dtype=torch.float32),
            torch.as_tensor(histories_np, device=self.device, dtype=torch.float32),
            torch.as_tensor(history_mask_np, device=self.device, dtype=torch.bool),
            dynamic_obstacles_np,
        )
