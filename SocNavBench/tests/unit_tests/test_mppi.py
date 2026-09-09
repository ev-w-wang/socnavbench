import os
import sys

import numpy as np

_SOCNAV_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(_SOCNAV_ROOT, "joystick"))

from joystick_py.joystick_mppi import (
    SocNavMPPIStateAdapter,
    build_mppi_controller,
    load_mppi_config,
)
from mppi import GPUMPPIController, MPPIController, torch


class _Config:
    def __init__(self, pose):
        self.pose = np.asarray(pose, dtype=float)

    def position_and_heading_nk3(self, squeeze=False):
        if squeeze:
            return self.pose
        return self.pose.reshape((1, 1, 3))


class _Agent:
    def __init__(self, pose, radius):
        self.config = _Config(pose)
        self.radius = radius

    def get_current_config(self):
        return self.config

    def get_radius(self):
        return self.radius


class _SimState:
    def __init__(self, robot, pedestrians, sim_t):
        self.robot = robot
        self.pedestrians = pedestrians
        self.sim_t = sim_t

    def get_robot(self):
        return self.robot

    def get_pedestrians(self):
        return self.pedestrians

    def get_sim_t(self):
        return self.sim_t


def _test_config_and_factory():
    config_path = os.path.join(
        _SOCNAV_ROOT,
        "policy.config",
    )
    config = load_mppi_config(config_path)
    assert config["dt"] == 0.05
    assert config["v_max"] == 1.2
    assert config["w_max"] == 1.1
    controller = build_mppi_controller(dict(config, use_gpu=False))
    assert isinstance(controller, MPPIController)


def _test_state_adapter_and_distance_field():
    traversible = np.ones((10, 10), dtype=bool)
    traversible[:, 0] = False
    environment = {"map_traversible": traversible, "map_scale": 0.1}
    adapter = SocNavMPPIStateAdapter(
        environment,
        robot_goal=[0.8, 0.8, 0.0],
        robot_radius=0.1,
    )
    first = _SimState(
        _Agent([0.5, 0.5, 0.0], 0.1),
        {"ped": _Agent([0.3, 0.4, 0.0], 0.2)},
        0.0,
    )
    second = _SimState(
        _Agent([0.5, 0.5, 0.0], 0.1),
        {"ped": _Agent([0.4, 0.4, 0.0], 0.2)},
        0.1,
    )
    state0 = adapter.adapt(first)
    state1 = adapter.adapt(second)
    assert np.allclose(state0.human_states[0].velocity, [0.0, 0.0])
    assert np.allclose(state1.human_states[0].velocity, [1.0, 0.0])
    assert state1.static_obstacle_map.clearance_at([0.05, 0.5]) < 0.0
    assert state1.static_obstacle_map.clearance_at([0.8, 0.5]) > 0.0


def _test_dynamics_and_action_bounds():
    cpu = MPPIController(
        horizon=3,
        num_iterations=1,
        num_samples=4,
        dt=0.05,
        v_max=1.2,
        w_min=-1.1,
        w_max=1.1,
        dynamic_prediction_method="constant_velocity",
    )
    state = np.array([1.0, 2.0, 0.5])
    action = np.array([1.2, -1.1])
    cpu_next = cpu.next_state_dynamics(state, action, 0.05)
    if torch is not None:
        gpu_next = GPUMPPIController._next_state_batch(
            torch.tensor(state[None], dtype=torch.float32),
            torch.tensor(action[None], dtype=torch.float32),
            0.05,
        ).cpu().numpy()[0]
        assert np.allclose(cpu_next, gpu_next, atol=1e-6)

    noise = np.full((4, 3, 2), 100.0)
    controls = cpu._build_noisy_control_samples(np.zeros((3, 2)), noise)
    assert np.all(controls[:, :, 0] <= 1.2)
    assert np.all(controls[:, :, 1] <= 1.1)


def _test_modular_predictions():
    if __import__("trajectory_predictors").rvo2 is None:
        raise AssertionError("rvo2 must be installed for MPPI evaluation")
    environment = {
        "map_traversible": np.ones((10, 10), dtype=bool),
        "map_scale": 0.1,
    }
    adapter = SocNavMPPIStateAdapter(
        environment,
        robot_goal=[0.8, 0.8, 0.0],
        robot_radius=0.1,
    )
    state = adapter.adapt(
        _SimState(
            _Agent([0.5, 0.5, 0.0], 0.1),
            {"ped": _Agent([0.3, 0.4, 0.0], 0.2)},
            0.0,
        )
    )
    dynamic, _, _ = MPPIController()._dynamic_obstacles_to_arrays(state)

    constant_velocity = MPPIController(
        horizon=3,
        dt=0.1,
        dynamic_prediction_method="constant_velocity",
    )
    predictions = constant_velocity.predict_dynamic_obstacle_poses(
        state,
        dynamic,
    )
    assert predictions.shape == (3, 1, 2)
    assert np.allclose(predictions[:, 0], [[0.3, 0.4]] * 3)

    orca = MPPIController(horizon=3, dynamic_prediction_method="orca")
    predictions = orca.predict_dynamic_obstacle_poses(state, dynamic)
    assert predictions.shape == (3, 1, 2)
    assert np.isfinite(predictions).all()

    state.sim_time = 1.0
    state.human_states[0].ground_truth_position_at = (
        lambda query_time: np.array([query_time, 2.0 * query_time])
    )
    ground_truth = MPPIController(
        horizon=3,
        dt=0.1,
        dynamic_prediction_method="ground_truth",
    )
    predictions = ground_truth.predict_dynamic_obstacle_poses(state, dynamic)
    assert np.allclose(
        predictions[:, 0],
        [[1.1, 2.2], [1.2, 2.4], [1.3, 2.6]],
    )


def _dummy_control_state(with_pedestrian=True):
    traversible = np.ones((20, 20), dtype=bool)
    traversible[:, 0] = False
    adapter = SocNavMPPIStateAdapter(
        {"map_traversible": traversible, "map_scale": 0.1},
        robot_goal=[1.5, 1.0, 0.0],
        robot_radius=0.1,
    )
    pedestrians = {}
    if with_pedestrian:
        pedestrians = {"ped": _Agent([0.8, 1.0, 0.0], 0.2)}
    return adapter.adapt(
        _SimState(_Agent([0.5, 1.0, 0.0], 0.1), pedestrians, 0.0)
    )


def _test_cuda_rollout():
    if torch is None or not torch.cuda.is_available():
        raise AssertionError("CUDA Torch is required to test GPU MPPI")
    assert "sm_86" in torch.cuda.get_arch_list()
    controller = GPUMPPIController(
        horizon=3,
        num_iterations=1,
        num_samples=8,
        dt=0.05,
        v_max=1.2,
        w_min=-1.1,
        w_max=1.1,
        device="cuda",
        compile_rollout=False,
        dynamic_prediction_method="constant_velocity",
    )
    assert controller.device.type == "cuda"
    action = controller.control_action(_dummy_control_state())
    assert action.shape == (2,)
    assert np.isfinite(action).all()
    assert -1e-5 <= action[0] <= 1.2 + 1e-5
    assert -1.1 - 1e-5 <= action[1] <= 1.1 + 1e-5


def _test_cpu_gpu_loss_match():
    if torch is None or not torch.cuda.is_available():
        raise AssertionError("CUDA Torch is required to test GPU MPPI")
    cpu = MPPIController(
        horizon=3,
        num_iterations=1,
        num_samples=4,
        dt=0.05,
        temp=10.0,
        dynamic_prediction_method="constant_velocity",
    )
    x_t = np.array([1.0, 2.0, 0.3], dtype=float)
    goal = np.array([4.0, 2.0], dtype=float)
    u_t = np.array([0.8, 0.1], dtype=float)
    x_prev = np.array([0.96, 2.0, 0.29], dtype=float)
    u_prev = np.array([0.7, 0.0], dtype=float)
    obstacles = np.array([[2.0, 2.0, 0.3]], dtype=float)
    dynamic = np.array([[3.0, 2.0, 0.4, 0.1, 0.0]], dtype=float)
    predictions = np.array([[[3.05, 2.0]], [[3.10, 2.0]], [[3.15, 2.0]]], dtype=float)
    histories = np.zeros((1, 20, 2), dtype=float)
    history_mask = np.zeros((1, 20), dtype=bool)
    cpu_loss = cpu.loss(
        x_t,
        goal,
        u_t,
        obstacles,
        dynamic,
        predictions,
        histories,
        history_mask,
        0,
        x_prev,
        u_prev,
        None,
    )
    gpu_loss = GPUMPPIController._loss_batch(
        torch.tensor(x_t[None], dtype=torch.float32, device="cuda"),
        torch.tensor(goal, dtype=torch.float32, device="cuda"),
        torch.tensor(u_t[None], dtype=torch.float32, device="cuda"),
        torch.tensor(obstacles, dtype=torch.float32, device="cuda"),
        torch.tensor(dynamic, dtype=torch.float32, device="cuda"),
        torch.tensor(predictions, dtype=torch.float32, device="cuda"),
        torch.tensor(histories, dtype=torch.float32, device="cuda"),
        torch.tensor(history_mask, dtype=torch.bool, device="cuda"),
        torch.empty((0, 0), dtype=torch.float32, device="cuda"),
        1.0,
        0.1,
        torch.tensor(cpu.std_diag, dtype=torch.float32, device="cuda"),
        cpu.temp,
        0,
        torch.tensor(x_prev[None], dtype=torch.float32, device="cuda"),
        torch.tensor(u_prev[None], dtype=torch.float32, device="cuda"),
        cpu.goal_radius,
        cpu.goal_weight,
        cpu.progress_weight,
        cpu.heading_weight,
        cpu.desired_speed,
        cpu.goal_speed_gain,
        cpu.slowdown_radius,
        cpu.speed_weight,
        cpu.stop_speed_weight,
        cpu.control_effort_weight,
        cpu.static_obs_clearance,
        cpu.static_obs_weight,
        cpu.dynamic_obs_clearance,
        cpu.dynamic_obs_weight,
        cpu.control_smoothing_weight,
    )
    assert np.allclose(cpu_loss, float(gpu_loss.cpu()), atol=1e-4)


def _test_gpu_factory():
    if torch is None or not torch.cuda.is_available():
        raise AssertionError("CUDA Torch is required to test GPU MPPI")
    config_path = os.path.join(_SOCNAV_ROOT, "policy.config")
    config = load_mppi_config(config_path)
    controller = build_mppi_controller(
        dict(config, use_gpu=True, compile_gpu_rollout=False)
    )
    assert isinstance(controller, GPUMPPIController)
    assert controller.device.type == "cuda"


def main_test():
    _test_config_and_factory()
    _test_state_adapter_and_distance_field()
    _test_dynamics_and_action_bounds()
    _test_modular_predictions()
    _test_cuda_rollout()
    _test_cpu_gpu_loss_match()
    _test_gpu_factory()
    print("MPPI tests passed")


if __name__ == "__main__":
    main_test()
