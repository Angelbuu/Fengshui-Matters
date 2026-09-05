"""OrcaLab backend for the Fengshui-Matters simulation adapter.

Orca-specific dependencies are loaded lazily so the normal project and test
suite do not require OrcaLab to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from src.navigation.schemas import Pose, RobotCommand, Velocity


@dataclass
class OrcaState:
    """Latest observation produced by OrcaLab."""

    run_id: str = "orca_run"
    command_id: str = "none"
    sim_time_s: float = 0.0
    robot_status: str = "STOPPED"

    pose: Pose = field(
        default_factory=lambda: Pose(x_m=0.0, y_m=0.0, yaw_rad=0.0)
    )
    velocity: Velocity = field(
        default_factory=lambda: Velocity(
            vx_mps=0.0,
            vy_mps=0.0,
            wz_rps=0.0,
        )
    )

    current_waypoint: str | None = None
    target_waypoint: str | None = None
    distance_to_target_m: float | None = None
    heading_error_rad: float = 0.0
    camera_frame: str | None = None
    obstacles: list[dict[str, Any]] = field(default_factory=list)



class OrcaSimulation:
    """Real OrcaLab implementation used by SimAdapter in Orca mode."""

    def __init__(
        self,
        orcagym_address: str | None = None,
        edit_address: str | None = None,
        lidar_entity: str = "LiDAR",
        robot_body: str = "quadruped_robot_1_base_link",
        camera_entity: str = "mujococamera1080",
        camera_wsl_dir: str | None = None,
    ) -> None:
        self.orcagym_address = orcagym_address or os.getenv(
            "ORCA_GYM_ADDRESS",
            "172.17.160.1:50051",
        )
        self.edit_address = edit_address or os.getenv(
            "ORCA_EDIT_ADDRESS",
            "172.17.160.1:50151",
        )
        self.lidar_entity = lidar_entity
        self.robot_body = robot_body
        self.camera_entity = camera_entity
        self.camera_wsl_dir = camera_wsl_dir or os.getenv(
            "ORCA_CAMERA_OUTPUT_DIR",
            "/tmp/fengshui_orca_camera",
        )

        self._state = OrcaState()
        self._connected = False
        self._channel = None
        self._stub = None
        self._mjc_message_pb2 = None
        self._np = None
        self._backend = None
        self._renderer = None
        self._last_physics_state = None

    def connect(self) -> None:
        """Connect to OrcaLab's gRPC service.

        Imports are intentionally local so importing Fengshui-Matters does not
        require the OrcaLab Python environment.
        """
        if self._connected:
            return

        try:
            import grpc
            import numpy as np
            from orca_gym.protos import mjc_message_pb2
            from orca_gym.protos import mjc_message_pb2_grpc
        except ImportError as exc:
            raise RuntimeError(
                "Orca mode requires the OrcaLab Python environment."
            ) from exc

        channel = grpc.insecure_channel(self.orcagym_address)
        grpc.channel_ready_future(channel).result(timeout=3.0)

        self._channel = channel
        self._stub = mjc_message_pb2_grpc.GrpcServiceStub(channel)
        self._mjc_message_pb2 = mjc_message_pb2
        self._np = np
        self._connected = True

    def get_lidar_obstacles(
        self,
        *,
        max_distance_m: float = 5.0,
        min_height_m: float = 0.15,
    ) -> list[dict[str, Any]]:
        """Return raw obstacle-height LiDAR detections.

        This does not plan a path or decide how Navigation should react.
        It only converts OrcaLab sensor data into simulator observations.
        """
        self.connect()

        np = self._np
        pb2 = self._mjc_message_pb2
        assert np is not None
        assert pb2 is not None
        assert self._stub is not None

        request = pb2.LiDARPointCloudRequest(
            entity_name=self.lidar_entity
        )
        response = self._stub.QueryLiDARPointCloud(
            request,
            timeout=2.0,
        )

        if response.status == pb2.LiDARPointCloudResponse.ENTITY_NOT_FOUND:
            raise RuntimeError(
                f"LiDAR entity not found: {self.lidar_entity}"
            )

        if response.status == pb2.LiDARPointCloudResponse.NO_DATA:
            self._state.obstacles = []
            return []

        ranges = np.frombuffer(
            response.range_data,
            dtype=np.float32,
        ).copy().reshape(
            response.bin_count,
            response.vertical_layers,
        )

        points = np.frombuffer(
            response.point_data,
            dtype=np.float32,
        ).copy().reshape(
            response.bin_count,
            response.vertical_layers,
            3,
        )

        valid = (
            np.isfinite(ranges)
            & (ranges > 0.15)
            & (ranges <= float(max_distance_m))
        )

        detected_ranges = ranges[valid]
        detected_points = points[valid]

        if len(detected_points) == 0:
            self._state.obstacles = []
            return []

        # Remove floor / very low returns.
        height_mask = detected_points[:, 2] > float(min_height_m)
        detected_ranges = detected_ranges[height_mask]
        detected_points = detected_points[height_mask]

        if len(detected_points) == 0:
            self._state.obstacles = []
            return []

        # For the first integration version expose the nearest sensed surface.
        # Clustering into multiple obstacles can be added after the interface
        # is proven end-to-end.
        index = int(np.argmin(detected_ranges))
        point = detected_points[index]
        distance = float(detected_ranges[index])

        # Convert OrcaLab world coordinates into the robot-relative frame.
        # Team convention: +x = forward, +y = left.
        import math

        pose = self.get_robot_pose()

        world_x = float(point[0])
        world_y = float(point[1])
        world_z = float(point[2])

        dx = world_x - pose.x_m
        dy = world_y - pose.y_m

        c = math.cos(pose.yaw_rad)
        s = math.sin(pose.yaw_rad)

        relative_x = c * dx + s * dy
        relative_y = -s * dx + c * dy

        obstacle = {
            "id": "lidar_nearest",
            "distance_m": distance,
            "relative_x_m": relative_x,
            "relative_y_m": relative_y,
            "world_x_m": world_x,
            "world_y_m": world_y,
            "world_z_m": world_z,
        }

        self._state.obstacles = [obstacle]
        return [obstacle]

    def _ensure_locomotion(self) -> None:
        """Start one persistent CPU Go2 policy and bind it to the authored scene."""
        if self._backend is not None and self._renderer is not None:
            return

        try:
            from functools import partial
            from navila_orca.backends.mjlab_go2 import MjlabGo2Backend
            from navila_orca.render.orca import OrcaLabRenderBridge
            from navila_orca.render.orca_camera import (
                OrcaMujocoCameraFollower,
                OrcaMujocoPngCamera,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Orca locomotion requires NaVILA-Orca/src on PYTHONPATH."
            ) from exc

        backend = MjlabGo2Backend(
            device="cpu",
            num_envs=1,
            deterministic_play=True,
            warmup_steps=100,
        )

        renderer = None

        try:
            # Renderer construction requires the backend's joint metadata.
            backend.start()

            renderer = OrcaLabRenderBridge(
                orcagym_address=self.orcagym_address,
                camera_port=7070,
                camera_name="navila",
                num_envs=1,
                joint_qpos_addr=backend.joint_qpos_addr,

                # Reuse the robot already authored in OrcaLab.
                agent_name=None,
                discover_agents=True,
                publish=False,
                anchor_to_scene=True,

                scene_profile="orca-train",
                strict_scene_options=True,
                manual_xml_override=True,
                aligned_xml_output=(
                    Path("outputs/orca/scene_alignment/aligned_scene.xml")
                ),

                # Let the same bridge own robot + camera synchronization.
                bind_camera=True,
                edit_address=self.edit_address,
                camera_actor_name=self.camera_entity,
                camera_asset_path="prefabs/mujococamera1080",
                camera_factory=partial(
                    OrcaMujocoPngCamera,
                    edit_address=self.edit_address,
                    remote_camera_name=self.camera_entity,
                    timeout_s=5.0,
                    output_dir=self.camera_wsl_dir,
                ),
                camera_follower_factory=OrcaMujocoCameraFollower,
            )

            # Establish the renderer's source-root reference while the local
            # backend is at its initial pose.
            state = backend.reset()
            renderer.start()
            renderer.push_state(state, backend.qpos_batch)

        except Exception:
            if renderer is not None:
                try:
                    renderer.close()
                except Exception:
                    pass
            try:
                backend.close()
            except Exception:
                pass
            raise

        self._backend = backend
        self._renderer = renderer
        self._last_physics_state = state

    def execute_command(self, command: RobotCommand) -> bool:
        """Execute one Navigation-Control velocity command on the simulated Go2."""
        import math

        self._ensure_locomotion()

        backend = self._backend
        renderer = self._renderer

        assert backend is not None
        assert renderer is not None

        self._state.run_id = command.run_id
        self._state.command_id = command.command_id
        self._state.target_waypoint = command.target_waypoint

        if command.action in {"STOP", "HOLD"}:
            vx = 0.0
            vy = 0.0
            wz = 0.0
        else:
            vx = float(command.vx_mps)
            vy = float(command.vy_mps)
            wz = float(command.wz_rps)

        self._state.velocity = Velocity(
            vx_mps=vx,
            vy_mps=vy,
            wz_rps=wz,
        )

        # STOP/HOLD still updates the command latch, but a zero-duration
        # command does not need to advance physics.
        backend.set_velocity_command(vx, vy, wz)

        duration = float(command.duration_s)
        ticks = (
            int(math.ceil(duration / backend.control_dt))
            if duration > 0.0
            else 0
        )

        try:
            for _ in range(ticks):
                step = backend.step()
                self._last_physics_state = step.state

                renderer.push_state(
                    step.state,
                    backend.qpos_batch,
                )

                if step.terminated or step.truncated:
                    self._state.robot_status = "COLLISION"
                    break
            else:
                self._state.robot_status = (
                    "STOPPED"
                    if command.action in {"STOP", "HOLD"}
                    else "RUNNING"
                )

            if self._last_physics_state is not None:
                self._state.sim_time_s = float(
                    self._last_physics_state.sim_time_s
                )

            # Read the actual visible OrcaLab pose after executing the chunk.
            self._state.pose = self.get_robot_pose()

            return True

        except Exception:
            self._state.robot_status = "ERROR"
            raise

        finally:
            # Never leave a motion command latched between Navigation calls.
            backend.set_velocity_command(0.0, 0.0, 0.0)

    def get_robot_pose(self) -> Pose:
        """Return the authoritative robot pose in OrcaLab world coordinates."""
        import math

        # Once locomotion is active, MJLab is the authoritative physics state.
        # Use the renderer's scene mapping so the local MJLab pose is expressed
        # in the authored OrcaLab world coordinate system.
        if self._backend is not None and self._renderer is not None:
            batch_renderer = getattr(self._renderer, "_renderer", None)

            if batch_renderer is not None:
                positions, quaternions = batch_renderer.map_root_pose(
                    self._backend.qpos_batch
                )

                position = positions[0]
                w, x, y, z = [float(v) for v in quaternions[0]]

                siny_cosp = 2.0 * (w * z + x * y)
                cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
                yaw = math.atan2(siny_cosp, cosy_cosp)

                pose = Pose(
                    x_m=float(position[0]),
                    y_m=float(position[1]),
                    yaw_rad=yaw,
                )

                self._state.pose = pose
                return pose

        # Before locomotion starts, query the authored OrcaLab robot directly.
        self.connect()

        pb2 = self._mjc_message_pb2
        assert pb2 is not None
        assert self._stub is not None

        response = self._stub.QueryBodyPosMatQuat(
            pb2.QueryBodyPosMatQuatRequest(
                body_name_list=[self.robot_body]
            ),
            timeout=2.0,
        )

        if not response.body_pos_mat_quat_list:
            raise RuntimeError(
                f"Robot body not found: {self.robot_body}"
            )

        body = response.body_pos_mat_quat_list[0]

        w, x, y, z = [float(v) for v in body.quat]

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        pose = Pose(
            x_m=float(body.pos[0]),
            y_m=float(body.pos[1]),
            yaw_rad=yaw,
        )

        self._state.pose = pose
        return pose

    def get_velocity(self) -> Velocity:
        return self._state.velocity

    def get_camera_frame(self) -> str | None:
        """Capture RGB through the synchronized OrcaLab render bridge."""
        import os

        self._ensure_locomotion()

        assert self._renderer is not None
        assert self._backend is not None
        assert self._last_physics_state is not None

        frame = self._renderer.capture(
            self._last_physics_state,
            self._backend.qpos_batch,
        )

        image = frame.rgb

        os.makedirs(self.camera_wsl_dir, exist_ok=True)

        output_path = os.path.join(
            self.camera_wsl_dir,
            f"fengshui_camera_{frame.frame_id.replace(':', '_')}.png",
        )

        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Saving Orca camera frames requires Pillow."
            ) from exc

        Image.fromarray(image).save(output_path)

        self._state.camera_frame = output_path
        return output_path

    def get_state(self) -> OrcaState:
        return self._state

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass

        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:
                pass

        self._renderer = None
        self._backend = None
        self._last_physics_state = None

        if self._channel is not None:
            self._channel.close()

        self._channel = None
        self._stub = None
        self._connected = False
