"""Simulation adapter for fake tests and the real OrcaLab backend."""

import time
import math

from control.adapters.base_adapter import BaseAdapter
from navigation.schemas import (
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
         # Lightweight robot state used only in fake mode.
        self._fake_x_m = 0.0
        self._fake_y_m = 0.0
        self._fake_yaw_rad = 0.0

        self._fake_vx_mps = 0.0
        self._fake_vy_mps = 0.0
        self._fake_wz_rps = 0.0

        self._fake_sim_time_s = 0.0

        if mode == "orca":
            # Lazy import keeps normal tests independent of OrcaLab/NaVILA.
            from src.control.adapters.orca_simulation import OrcaSimulation

            self._orca = OrcaSimulation()

    def send_command(self, command: RobotCommand) -> bool:
        self.last_command_received = command

        if self.mode == "fake":

            duration = command.duration_s

            # Save commanded velocity for the next observation.
            self._fake_vx_mps = command.vx_mps
            self._fake_vy_mps = command.vy_mps
            self._fake_wz_rps = command.wz_rps

            # ----------------------------------------
            # Update orientation
            # ----------------------------------------

            self._fake_yaw_rad += (
                command.wz_rps * duration
            )

            # Keep yaw inside [-pi, pi].
            self._fake_yaw_rad = (
                self._fake_yaw_rad + math.pi
            ) % (2 * math.pi) - math.pi

            # ----------------------------------------
            # Update position
            # ----------------------------------------
            #
            # vx / vy are robot-local velocities.
            # Convert them into world coordinates.
            # ----------------------------------------

            cos_yaw = math.cos(self._fake_yaw_rad)
            sin_yaw = math.sin(self._fake_yaw_rad)

            world_vx = (
                command.vx_mps * cos_yaw
                - command.vy_mps * sin_yaw
            )

            world_vy = (
                command.vx_mps * sin_yaw
                + command.vy_mps * cos_yaw
            )

            self._fake_x_m += world_vx * duration
            self._fake_y_m += world_vy * duration

            self._fake_sim_time_s += duration

            print(
                f"[SimAdapter] Sent command: "
                f"{command.action} "
                f"for {duration:.2f}s"
            )

            return True

        assert self._orca is not None
        return self._orca.execute_command(command)

    def get_observation(self) -> SimulatorObservation:
        if self.mode == "fake":

            return SimulatorObservation(
                run_id="fake_run",
                command_id=(
                    self.last_command_received.command_id
                    if self.last_command_received
                    else "fake_cmd"
                ),
                timestamp_ms=int(time.time() * 1000),
                sim_time_s=self._fake_sim_time_s,
                robot_status="RUNNING",

                pose=Pose(
                    x_m=self._fake_x_m,
                    y_m=self._fake_y_m,
                    yaw_rad=self._fake_yaw_rad,
                ),

                velocity=Velocity(
                    vx_mps=self._fake_vx_mps,
                    vy_mps=self._fake_vy_mps,
                    wz_rps=self._fake_wz_rps,
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
