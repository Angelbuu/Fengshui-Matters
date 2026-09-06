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
        robot_body: str = "go2_000_base_link",
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

        # Current production navigation uses LiDAR only.
        self.camera_enabled = False

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

    def get_runtime_obstacles(
        self,
        *,
        max_forward_m: float = 0.50,
        half_width_m: float = 0.32,
    ) -> list[dict[str, Any]]:
        """Detect generic collidable MuJoCo geometry ahead of the Go2.

        This is simulator-side proximity sensing for the safety layer.
        It does not depend on obstacle names, destination names, or
        hard-coded obstacle coordinates.
        """
        import math

        self.connect()

        pb2 = self._mjc_message_pb2

        assert pb2 is not None
        assert self._stub is not None

        pose = self.get_robot_pose()

        # Discover all collision geoms in the current runtime model.
        all_geoms = self._stub.QueryAllGeoms(
            pb2.QueryAllGeomsRequest(),
            timeout=2.0,
        )

        candidates = []

        for geom in all_geoms.geom_data_list:

            name = geom.geom_name
            body = geom.body_name

            # Never detect the runtime Go2 as its own obstacle.
            if (
                body.startswith("go2_000")
                or name.startswith("go2_000")
            ):
                continue

            # Ignore visual-only/non-collidable geometry.
            if (
                int(geom.geom_contype) == 0
                and int(geom.geom_conaffinity) == 0
            ):
                continue

            # MuJoCo geom type 0 is a plane. Ground planes are
            # intentionally collidable so the robot can stand on
            # them, but they are not navigation obstacles.
            if int(geom.geom_type) == 0:
                continue

            candidates.append(geom)

        if not candidates:
            self._state.obstacles = []
            return []

        names = [
            geom.geom_name
            for geom in candidates
        ]

        positions = self._stub.QueryGeomPosMat(
            pb2.QueryGeomPosMatRequest(
                geom_name_list=names,
            ),
            timeout=2.0,
        )

        position_by_name = {
            item.geom_name: item
            for item in positions.geom_pos_mat_list
        }

        c = math.cos(pose.yaw_rad)
        s = math.sin(pose.yaw_rad)

        obstacles = []

        for geom in candidates:

            current = position_by_name.get(
                geom.geom_name
            )

            if current is None:
                continue

            wx = float(current.pos[0])
            wy = float(current.pos[1])

            dx = wx - pose.x_m
            dy = wy - pose.y_m

            # World -> robot coordinates.
            forward = c * dx + s * dy
            lateral = -s * dx + c * dy

            size = list(geom.geom_size)

            # Conservative XY radius around the geom centre.
            #
            # This works for boxes, cylinders and other common
            # prototype obstacles without depending on their names.
            radius = (
                max(float(size[0]), float(size[1]))
                if len(size) >= 2
                else float(size[0])
                if size
                else 0.0
            )

            nearest_forward = forward - radius
            lateral_clearance = (
                abs(lateral) - radius
            )

            # Obstacle footprint overlaps the robot's forward
            # collision corridor.
            blocking = (
                forward > 0.0
                and nearest_forward
                <= max_forward_m
                and lateral_clearance
                <= half_width_m
            )

            if not blocking:
                continue

            distance = max(
                0.0,
                math.hypot(forward, lateral)
                - radius,
            )

            obstacles.append(
                {
                    "id": geom.geom_name,
                    "name": geom.geom_name,
                    "body": geom.body_name,
                    "distance_m": distance,
                    "relative_x_m": forward,
                    "relative_y_m": lateral,
                    "blocking": True,
                    "is_blocking": True,
                    "source": "mujoco_runtime",
                }
            )

        obstacles.sort(
            key=lambda item: item["distance_m"]
        )

        self._state.obstacles = obstacles

        return obstacles


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

    def get_forward_danger_obstacle(
        self,
        *,
        max_forward_m: float = 0.60,
        half_width_m: float = 0.25,
        min_height_m: float = 0.15,
    ):
        """
        Return the nearest LiDAR point inside a narrow collision
        corridor directly in front of the Go2.

        Unlike get_lidar_obstacles(), this examines ALL valid
        LiDAR returns instead of only the globally nearest surface.
        """
        import math

        self.connect()

        np = self._np
        pb2 = self._mjc_message_pb2

        assert np is not None
        assert pb2 is not None
        assert self._stub is not None

        response = self._stub.QueryLiDARPointCloud(
            pb2.LiDARPointCloudRequest(
                entity_name=self.lidar_entity
            ),
            timeout=2.0,
        )

        if response.status != pb2.LiDARPointCloudResponse.SUCCESS:
            return None

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
            & (ranges <= 2.0)
        )

        detected_ranges = ranges[valid]
        detected_points = points[valid]

        if len(detected_points) == 0:
            return None

        height_mask = (
            detected_points[:, 2]
            > float(min_height_m)
        )

        detected_ranges = detected_ranges[height_mask]
        detected_points = detected_points[height_mask]

        if len(detected_points) == 0:
            return None

        pose = self.get_robot_pose()

        c = math.cos(pose.yaw_rad)
        s = math.sin(pose.yaw_rad)

        best = None

        for distance, point in zip(
            detected_ranges,
            detected_points,
        ):
            # Safety-stop only for a return that is genuinely
            # physically close according to the LiDAR itself.
            # This rejects geometry/self returns whose transformed
            # robot-relative position appears close even though
            # their actual sensor range is much farther away.
            if float(distance) > 0.30:
                continue

            world_x = float(point[0])
            world_y = float(point[1])

            dx = world_x - pose.x_m
            dy = world_y - pose.y_m

            relative_x = c * dx + s * dy
            relative_y = -s * dx + c * dy

            # Ignore points behind/inside the robot.
            # Ignore near-origin/self returns from the Go2.
            if relative_x <= 0.10:
                continue

            # Ignore anything farther than the emergency zone.
            if relative_x >= max_forward_m:
                continue

            # Ignore walls/objects beside the robot.
            if abs(relative_y) >= half_width_m:
                continue

            candidate = {
                "distance_m": float(distance),
                "relative_x_m": relative_x,
                "relative_y_m": relative_y,
                "world_x_m": world_x,
                "world_y_m": world_y,
                "world_z_m": float(point[2]),
            }

            if (
                best is None
                or relative_x
                < best["relative_x_m"]
            ):
                best = candidate

        return best


    def get_lidar_clearances(
        self,
        *,
        max_distance_m: float = 5.0,
        min_height_m: float = 0.15,
    ) -> dict[str, float]:
        """Return directional obstacle clearances in the robot frame.

        Sectors:
        front, front_left, front_right, left, right.

        Values are distances in metres. A sector with no detected
        obstacle returns max_distance_m.
        """
        import math

        self.connect()

        np = self._np
        pb2 = self._mjc_message_pb2

        assert np is not None
        assert pb2 is not None
        assert self._stub is not None

        response = self._stub.QueryLiDARPointCloud(
            pb2.LiDARPointCloudRequest(
                entity_name=self.lidar_entity
            ),
            timeout=2.0,
        )

        result = {
            "front": float(max_distance_m),
            "front_left": float(max_distance_m),
            "front_right": float(max_distance_m),
            "left": float(max_distance_m),
            "right": float(max_distance_m),
        }

        if response.status != pb2.LiDARPointCloudResponse.SUCCESS:
            return result

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
            return result

        height_mask = (
            detected_points[:, 2] > float(min_height_m)
        )

        detected_ranges = detected_ranges[height_mask]
        detected_points = detected_points[height_mask]

        if len(detected_points) == 0:
            return result

        pose = self.get_robot_pose()

        c = math.cos(pose.yaw_rad)
        s = math.sin(pose.yaw_rad)

        for distance, point in zip(
            detected_ranges,
            detected_points,
        ):
            dx = float(point[0]) - pose.x_m
            dy = float(point[1]) - pose.y_m

            # World -> robot coordinates.
            relative_x = c * dx + s * dy
            relative_y = -s * dx + c * dy

            angle = math.degrees(
                math.atan2(relative_y, relative_x)
            )

            d = float(distance)

            # Collision corridor directly ahead.
            # Use robot-relative lateral distance so obstacles
            # that overlap the physical width of the Go2 count
            # as front obstacles.
            if (
                relative_x > 0.05
                and abs(relative_y) <= 0.20
            ):
                if relative_x < result["front"]:
                    result["front"] = relative_x
                    result["front_debug"] = {
                        "relative_x": relative_x,
                        "relative_y": relative_y,
                        "world_x": float(point[0]),
                        "world_y": float(point[1]),
                        "range": d,
                    }

            # Directional sectors for choosing the clearer side.
            if 10.0 < angle <= 55.0:
                result["front_left"] = min(
                    result["front_left"],
                    d,
                )

            elif -55.0 <= angle < -10.0:
                result["front_right"] = min(
                    result["front_right"],
                    d,
                )

            elif 55.0 < angle <= 120.0:
                result["left"] = min(
                    result["left"],
                    d,
                )

            elif -120.0 <= angle < -55.0:
                result["right"] = min(
                    result["right"],
                    d,
                )

        return result

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

                # Publish the real runtime Go2 while preserving
                # the authored hospital environment.
                agent_name="go2_000",
                discover_agents=False,
                publish=True,
                preserve_scene=True,

                # Keep the runtime robot in the same ORCA world
                # coordinate system used by our destinations.
                spawn_center=(-0.710, -0.735, 0.0),
                spawn_height=0.30,
                anchor_to_scene=False,

                scene_profile="orca-train",
                strict_scene_options=True,
                manual_xml_override=True,
                aligned_xml_output=(
                    Path("outputs/orca/scene_alignment/aligned_scene.xml")
                ),

                # LiDAR-only navigation test. Re-enable RGB after
                # runtime locomotion is fully integrated.
                bind_camera=False,
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
        """Return the authoritative runtime Go2 pose in OrcaLab coordinates."""
        import math

        # go2_000 is created by the locomotion renderer, so ensure the
        # runtime robot exists before attempting to read its pose.
        if self._backend is None or self._renderer is None:
            self._ensure_locomotion()

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
