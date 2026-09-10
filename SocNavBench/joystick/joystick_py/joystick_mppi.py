import configparser
import os

import numpy as np
import scipy.interpolate

from joystick_py.joystick_base import JoystickBase
from mppi import (
    GPUMPPIController,
    MPPIController,
    OccupancyGridDistanceField,
    torch,
)
from params.central_params import (
    create_socnav_params,
    create_test_params,
    get_path_to_socnav,
    get_seed,
)


_BOOL_KEYS = {"use_gpu", "compile_gpu_rollout"}
_INT_KEYS = {
    "horizon",
    "num_iterations",
    "num_samples",
    "dynamic_history_len",
    "dynamic_time_smear_steps",
    "orca_max_neighbors",
}
_STRING_KEYS = {
    "gpu_device",
    "initial_sequence",
    "dynamic_prediction_method",
}
_CONTROL_KEYS = _BOOL_KEYS | _INT_KEYS | _STRING_KEYS


def load_mppi_config(path, section="mppi"):
    parser = configparser.ConfigParser()
    if not parser.read(path):
        raise IOError("Unable to read MPPI config: {}".format(path))
    if not parser.has_section(section):
        raise ValueError("Missing MPPI config section: {}".format(section))

    config = {}
    for key, value in parser.items(section):
        if key in _BOOL_KEYS:
            config[key] = parser.getboolean(section, key)
        elif key in _INT_KEYS:
            config[key] = parser.getint(section, key)
        elif key in _STRING_KEYS:
            config[key] = value
        elif key == "seed":
            config[key] = parser.getint(section, key)
        else:
            try:
                config[key] = parser.getfloat(section, key)
            except ValueError:
                raise ValueError(
                    "Unsupported MPPI config value {}={!r}".format(key, value)
                )
    return config


def build_mppi_controller(config):
    config = dict(config)
    use_gpu = config.pop("use_gpu", False)
    device_name = config.pop("gpu_device", "auto")
    compile_rollout = config.pop("compile_gpu_rollout", False)
    seed = config.pop("seed", get_seed())
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)

    if device_name == "auto":
        device_name = (
            "cuda"
            if torch is not None and torch.cuda.is_available()
            else "cpu"
        )
    cuda_available = torch is not None and torch.cuda.is_available()
    if use_gpu and device_name.startswith("cuda") and cuda_available:
        torch.cuda.manual_seed_all(seed)
        return GPUMPPIController(
            device=device_name,
            compile_rollout=compile_rollout,
            **config
        )
    if use_gpu:
        print("CUDA MPPI unavailable; falling back to NumPy CPU rollouts")
    return MPPIController(**config)


class _SelfState:
    pass


class _HumanState:
    pass


class _MPPIState:
    pass


def load_ground_truth_sources(episode_name):
    """Reconstruct prerecorded trajectories for oracle evaluation."""
    from agents.humans.recorded_human import PrerecordedHuman

    params = create_socnav_params()
    episode = create_test_params(episode_name)
    sources = {}
    for index, dataset in enumerate(episode.pedestrian_datasets):
        humans = PrerecordedHuman.generate_humans(
            params,
            max_time=episode.max_time,
            start_t=episode.datasets_start_t[index],
            ped_range=episode.ped_ranges[index],
            dataset=dataset,
        )
        for human in humans:
            sources[human.get_name()] = human
    return sources


class SocNavMPPIStateAdapter:
    def __init__(
        self,
        environment,
        robot_goal,
        robot_radius,
        ground_truth_sources=None,
    ):
        traversible = np.asarray(environment["map_traversible"], dtype=bool)
        resolution = float(environment["map_scale"])
        self.static_obstacle_map = OccupancyGridDistanceField(
            traversible,
            resolution,
            robot_radius,
        )
        self.robot_goal = np.asarray(robot_goal, dtype=float)
        self.robot_radius = float(robot_radius)
        self.ground_truth_sources = ground_truth_sources or {}
        self._previous_humans = {}
        self._previous_time = None

    def reset(self):
        self._previous_humans = {}
        self._previous_time = None

    def adapt(self, sim_state):
        robot = sim_state.get_robot()
        robot_pose = robot.get_current_config().position_and_heading_nk3(
            squeeze=True
        )
        self_state = _SelfState()
        self_state.px = float(robot_pose[0])
        self_state.py = float(robot_pose[1])
        self_state.theta = float(robot_pose[2])
        self_state.gx = float(self.robot_goal[0])
        self_state.gy = float(self.robot_goal[1])
        self_state.radius = self.robot_radius

        current_time = float(sim_state.get_sim_t())
        elapsed = (
            current_time - self._previous_time
            if self._previous_time is not None
            else 0.0
        )
        current_humans = {}
        human_states = []
        for name, pedestrian in sim_state.get_pedestrians().items():
            pose = pedestrian.get_current_config().position_and_heading_nk3(
                squeeze=True
            )
            position = np.asarray(pose[:2], dtype=float)
            previous = self._previous_humans.get(name)
            if previous is None or elapsed <= 0.0:
                velocity = np.zeros(2, dtype=float)
            else:
                velocity = (position - previous) / elapsed
            human_state = _HumanState()
            human_state.position = position
            human_state.velocity = velocity
            human_state.radius = float(
                pedestrian.get_radius()
                if pedestrian.get_radius() is not None
                else 0.2
            )
            ground_truth_source = self.ground_truth_sources.get(name)
            if ground_truth_source is not None:
                human_state.ground_truth_position_at = (
                    lambda query_time, source=ground_truth_source: np.array(
                        (
                            source.xinterp(query_time),
                            source.yinterp(query_time),
                        ),
                        dtype=float,
                    )
                )
            human_states.append(human_state)
            current_humans[name] = position

        self._previous_humans = current_humans
        self._previous_time = current_time
        state = _MPPIState()
        state.self_state = self_state
        state.human_states = human_states
        state.static_obstacle_map = self.static_obstacle_map
        state.sim_time = current_time
        return state


class JoystickMPPI(JoystickBase):
    def __init__(self, config_section="mppi"):
        self.config_section = config_section
        self.controller = None
        self.state_adapter = None
        self.mppi_state = None
        self.command = [(0.0, 0.0)]
        self._first_control = True
        super().__init__("MPPI")

    def init_control_pipeline(self):
        if not self.joystick_params.use_system_dynamics:
            raise ValueError(
                "MPPI requires joystick_params.use_system_dynamics=True"
            )
        config_path = os.path.join(get_path_to_socnav(), "policy.config")
        config = load_mppi_config(config_path, self.config_section)
        self.controller = build_mppi_controller(config)
        environment = self.current_ep.get_environment()
        robot_radius = float(
            create_robot_radius()
        )
        ground_truth_sources = None
        if config["dynamic_prediction_method"] == "ground_truth":
            ground_truth_sources = load_ground_truth_sources(
                self.current_ep.get_name()
            )
        self.state_adapter = SocNavMPPIStateAdapter(
            environment,
            self.current_ep.get_robot_goal(),
            robot_radius,
            ground_truth_sources=ground_truth_sources,
        )
        self._first_control = True

    def joystick_sense(self):
        self.send_to_robot("sense")
        self.joystick_on = self.listen_once()
        if self.joystick_on:
            self.mppi_state = self.state_adapter.adapt(self.sim_state_now)

    def joystick_plan(self):
        if not self.joystick_on:
            return
        if self._first_control:
            self.controller.reset_cached_control_sequence(
                self.mppi_state.self_state
            )
            self._first_control = False
        action = self.controller.control_action(self.mppi_state)
        self.command = [(float(action[0]), float(action[1]))]

    def joystick_act(self):
        if self.joystick_on:
            self.send_cmds(self.command, send_vel_cmds=True)

    def update_loop(self):
        super().pre_update()
        while self.joystick_on:
            self.joystick_sense()
            self.joystick_plan()
            self.joystick_act()
        self.finish_episode()


def create_robot_radius():
    from params.central_params import create_robot_params

    return create_robot_params().physical_params.radius
