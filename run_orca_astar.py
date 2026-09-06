import argparse
import math
import uuid

from src.control.adapters.sim_adapter import SimAdapter
from src.navigation.orca_map import (
    DESTINATIONS,
    astar_path,
    build_occupancy_map,
    simplify_path,
)
from src.navigation.schemas import RobotCommand


WAYPOINT_REACHED_M = 0.25

HEADING_TOLERANCE_DEG = 10.0

TURN_SPEED_RPS = 0.45
TURN_DURATION_S = 0.30

MOVE_SPEED_MPS = 0.18
MOVE_DURATION_S = 0.40

# Emergency LiDAR brake only.
EMERGENCY_STOP_M = 0.55


def wrap_angle(angle):
    return (
        (angle + math.pi)
        % (2.0 * math.pi)
        - math.pi
    )


def command(
    obs,
    action,
    vx,
    wz,
    duration,
    target,
):
    return RobotCommand(
        run_id=obs.run_id,
        command_id=str(uuid.uuid4()),
        action=action,
        vx_mps=vx,
        vy_mps=0.0,
        wz_rps=wz,
        duration_s=duration,
        target_waypoint=target,
        safety_checked=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "destination",
        choices=DESTINATIONS.keys(),
    )

    parser.add_argument(
        "--max-decisions",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    destination = args.destination

    goal_x, goal_y = (
        DESTINATIONS[destination]
    )

    print("=" * 65)
    print("ORCA A* NAVIGATION")
    print("Destination:", destination)
    print(
        f"Destination marker: "
        f"({goal_x:.3f}, {goal_y:.3f})"
    )
    print("=" * 65)

    adapter = SimAdapter(mode="orca")

    try:
        # Publish/start the visible runtime Go2.
        adapter._orca._ensure_locomotion()

        batch = adapter._orca._renderer._renderer

        nav_map = build_occupancy_map(
            batch.combined_model,
            batch.agent_names,
        )

        obs = adapter.get_observation()

        path = astar_path(
            nav_map,
            obs.pose.x_m,
            obs.pose.y_m,
            goal_x,
            goal_y,
        )

        if path is None:
            raise RuntimeError(
                "A* could not find a route."
            )

        waypoints = simplify_path(
            path,
            spacing_m=0.50,
        )

        # First point is effectively current position.
        if len(waypoints) > 1:
            waypoints = waypoints[1:]

        print()
        print(
            f"Planned {len(waypoints)} "
            "physical waypoints:"
        )

        for i, (x, y) in enumerate(
            waypoints
        ):
            print(
                f"  {i:02d}: "
                f"({x:.2f}, {y:.2f})"
            )

        print()

        waypoint_index = 0
        emergency_stops = 0

        for decision in range(
            args.max_decisions
        ):
            obs = adapter.get_observation()

            if waypoint_index >= len(
                waypoints
            ):
                print()
                print("=" * 65)
                print(
                    f"ARRIVED AT "
                    f"{destination.upper()}"
                )
                print(
                    f"position="
                    f"({obs.pose.x_m:.3f}, "
                    f"{obs.pose.y_m:.3f})"
                )
                print("=" * 65)
                return

            target_x, target_y = (
                waypoints[waypoint_index]
            )

            dx = target_x - obs.pose.x_m
            dy = target_y - obs.pose.y_m

            distance = math.hypot(
                dx,
                dy,
            )

            # ----------------------------------------
            # Waypoint reached
            # ----------------------------------------

            if distance <= WAYPOINT_REACHED_M:

                print(
                    f"      WAYPOINT "
                    f"{waypoint_index} REACHED"
                )

                waypoint_index += 1
                continue

            desired_yaw = math.atan2(
                dy,
                dx,
            )

            heading_error = wrap_angle(
                desired_yaw
                - obs.pose.yaw_rad
            )

            heading_deg = math.degrees(
                heading_error
            )

            # ----------------------------------------
            # LiDAR emergency safety only
            # ----------------------------------------

            lidar = (
                adapter._orca
                .get_lidar_clearances()
            )

            front = lidar.get(
                "front",
                5.0,
            )

            # Orca can expose zero/near-zero LiDAR values for
            # invalid or missing returns. Do not interpret those
            # as a physical obstacle touching the robot.
            if front <= 0.05:
                front = 5.0

            if front < EMERGENCY_STOP_M:

                emergency_stops += 1

                if emergency_stops >= 5:
                    raise RuntimeError(
                        "Persistent LiDAR obstacle on A* path."
                    )

                cmd = command(
                    obs,
                    "STOP",
                    0.0,
                    0.0,
                    0.10,
                    destination,
                )

                action = "EMERGENCY_STOP"

            # ----------------------------------------
            # Turn toward A* waypoint
            # ----------------------------------------

            elif (
                abs(heading_deg)
                > HEADING_TOLERANCE_DEG
            ):

                emergency_stops = 0

                wz = (
                    TURN_SPEED_RPS
                    if heading_error > 0
                    else -TURN_SPEED_RPS
                )

                cmd = command(
                    obs,
                    "TURN",
                    0.0,
                    wz,
                    TURN_DURATION_S,
                    destination,
                )

                action = (
                    "TURN_LEFT"
                    if wz > 0
                    else "TURN_RIGHT"
                )

            # ----------------------------------------
            # Follow A* path
            # ----------------------------------------

            else:

                emergency_stops = 0

                cmd = command(
                    obs,
                    "MOVE",
                    MOVE_SPEED_MPS,
                    0.0,
                    MOVE_DURATION_S,
                    destination,
                )

                action = "MOVE"

            print(
                f"[{decision:03d}] "
                f"wp={waypoint_index:02d} "
                f"pos=({obs.pose.x_m:.2f},"
                f"{obs.pose.y_m:.2f}) "
                f"target=({target_x:.2f},"
                f"{target_y:.2f}) "
                f"dist={distance:.2f} "
                f"heading={heading_deg:.1f} "
                f"front={front:.2f} "
                f"action={action}"
            )

            adapter.send_command(cmd)

        print()
        print(
            "MAX DECISIONS EXCEEDED"
        )

    finally:
        adapter.close()


if __name__ == "__main__":
    main()
