"""Simulation adapter for fake tests and the real OrcaLab backend."""

import time

from src.control.adapters.base_adapter import BaseAdapter
from src.navigation.schemas import (
    Pose,
    RobotCommand,
    SimulatorObservation,
    Velocity,
)


class SimAdapter(BaseAdapter):
    """Navigation-to-simulation adapter.

    ``mode="fake"`` preserves the lightweight test behaviour and requires no
    Orca dependencies.

    ``mode="orca"`` connects Navigation Control to the real OrcaLab backend.
    """

    def __init__(self, mode: str = "fake"):
        if mode not in {"fake", "orca"}:
            raise ValueError("mode must be 'fake' or 'orca'")

        self.mode = mode
        self.last_command_received = None
        self._orca = None

        if mode == "orca":
            # Lazy import keeps normal tests independent of OrcaLab/NaVILA.
            from src.control.adapters.orca_simulation import OrcaSimulation

            self._orca = OrcaSimulation()

    def send_command(self, command: RobotCommand) -> bool:
        self.last_command_received = command

        if self.mode == "fake":
            print(
                f"[SimAdapter] Sent command to simulator: "
                f"{command.action} for {command.duration_s} seconds"
            )
            return True

        assert self._orca is not None
        return self._orca.execute_command(command)

    def get_observation(self) -> SimulatorObservation:
        if self.mode == "fake":
            return SimulatorObservation(
                run_id="fake_run",
                command_id="fake_cmd",
                timestamp_ms=int(time.time() * 1000),
                sim_time_s=0.0,
                robot_status="RUNNING",
                pose=Pose(
                    x_m=0.0,
                    y_m=0.0,
                    yaw_rad=0.0,
                ),
                velocity=Velocity(
                    vx_mps=0.0,
                    vy_mps=0.0,
                    wz_rps=0.0,
                ),
                heading_error_rad=0.0,
                obstacles=[],
            )

        assert self._orca is not None

        # Refresh all simulator-side observations.
        pose = self._orca.get_robot_pose()
        obstacles = self._orca.get_lidar_obstacles()
        camera_frame = self._orca.get_camera_frame()

        state = self._orca.get_state()

        return SimulatorObservation(
            run_id=state.run_id,
            command_id=state.command_id,
            timestamp_ms=int(time.time() * 1000),
            sim_time_s=state.sim_time_s,
            robot_status=state.robot_status,
            pose=pose,
            velocity=state.velocity,
            current_waypoint=state.current_waypoint,
            target_waypoint=state.target_waypoint,
            distance_to_target_m=state.distance_to_target_m,
            heading_error_rad=state.heading_error_rad,
            camera_frame=camera_frame,
            obstacles=obstacles,
        )

    def close(self) -> None:
        """Release Orca resources when running in real simulation mode."""
        if self._orca is not None:
            self._orca.close()
            self._orca = None
