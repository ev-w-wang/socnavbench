import numpy as np

try:
    import rvo2
except ImportError:
    rvo2 = None


class TrajectoryPredictor:
    """Interface for pedestrian trajectory predictors used by MPPI."""

    def __init__(self, horizon, dt):
        self.horizon = int(horizon)
        self.dt = float(dt)

    def predict(self, state, dynamic_obstacles):
        raise NotImplementedError


class ConstantVelocityPredictor(TrajectoryPredictor):
    def predict(self, state, dynamic_obstacles):
        if dynamic_obstacles.shape[0] == 0:
            return np.empty((self.horizon, 0, 2), dtype=float)
        timesteps = (
            np.arange(self.horizon, dtype=float) + 1.0
        ).reshape((-1, 1, 1))
        positions = dynamic_obstacles[:, 0:2]
        velocities = dynamic_obstacles[:, 3:5]
        return (
            positions[np.newaxis, :, :]
            + velocities[np.newaxis, :, :] * timesteps * self.dt
        )


class ORCAPredictor(TrajectoryPredictor):
    def __init__(
        self,
        horizon,
        dt,
        neighbor_dist,
        max_neighbors,
        time_horizon,
        time_horizon_obst,
        max_speed,
    ):
        super().__init__(horizon, dt)
        self.params = (
            float(neighbor_dist),
            int(max_neighbors),
            float(time_horizon),
            float(time_horizon_obst),
        )
        self.max_speed = max(float(max_speed), 1.0)

    def predict(self, state, dynamic_obstacles):
        if rvo2 is None:
            raise ImportError(
                "rvo2 is required for dynamic_prediction_method='orca'"
            )
        human_states = state.human_states
        if len(human_states) == 0:
            return np.empty((self.horizon, 0, 2), dtype=float)

        sim = rvo2.PyRVOSimulator(
            self.dt,
            *self.params,
            0.3,
            self.max_speed
        )
        for human_state in human_states:
            speed = np.linalg.norm(human_state.velocity)
            sim.addAgent(
                tuple(human_state.position),
                *self.params,
                human_state.radius,
                max(speed, self.max_speed),
                tuple(human_state.velocity),
            )

        predictions = np.empty(
            (self.horizon, len(human_states), 2),
            dtype=float,
        )
        for timestep in range(self.horizon):
            for index, human_state in enumerate(human_states):
                preferred_velocity = np.asarray(
                    human_state.velocity,
                    dtype=float,
                )
                speed = np.linalg.norm(preferred_velocity)
                if speed > self.max_speed:
                    preferred_velocity *= self.max_speed / speed
                sim.setAgentPrefVelocity(
                    index,
                    tuple(preferred_velocity),
                )
            sim.doStep()
            for index in range(len(human_states)):
                predictions[timestep, index] = sim.getAgentPosition(index)
        return predictions


class GroundTruthPredictor(TrajectoryPredictor):
    """Oracle predictor backed by prerecorded SocNavBench trajectories."""

    def predict(self, state, dynamic_obstacles):
        human_states = state.human_states
        if len(human_states) == 0:
            return np.empty((self.horizon, 0, 2), dtype=float)
        predictions = np.empty(
            (self.horizon, len(human_states), 2),
            dtype=float,
        )
        for index, human_state in enumerate(human_states):
            position_at = getattr(
                human_state,
                "ground_truth_position_at",
                None,
            )
            if position_at is None:
                raise ValueError(
                    "Ground-truth prediction requested for pedestrian "
                    "without a prerecorded trajectory"
                )
            for timestep in range(self.horizon):
                future_time = (
                    state.sim_time + (timestep + 1) * self.dt
                )
                predictions[timestep, index] = position_at(future_time)
        return predictions


_PREDICTOR_FACTORIES = {
    "constant_velocity": ConstantVelocityPredictor,
    "orca": ORCAPredictor,
    "ground_truth": GroundTruthPredictor,
}


def register_trajectory_predictor(name, factory):
    """Register a future learned or analytical predictor factory."""
    if not name:
        raise ValueError("Predictor name must be non-empty")
    _PREDICTOR_FACTORIES[name] = factory


def make_trajectory_predictor(method, horizon, dt, **kwargs):
    try:
        factory = _PREDICTOR_FACTORIES[method]
    except KeyError:
        raise ValueError(
            "Unknown dynamic obstacle prediction method: {}".format(method)
        )
    if method == "orca":
        return factory(horizon=horizon, dt=dt, **kwargs)
    return factory(horizon=horizon, dt=dt)
