import math
import uuid
from typing import Tuple

from navigation.schemas import RobotCommand
from navigation.schemas import RobotCommand, SimulatorObservation
from route_planner import DESTINATION_COORDINATES
from control.adapters.sim_adapter import SimAdapter
ARRIVAL_DISTANCE_M = 0.35
HEADING_TOLERANCE_DEG = 12.0

FORWARD_SPEED_MPS = 0.15
TURN_SPEED_RPS = 0.35

COMMAND_DURATION_S = 0.35
def wrap_angle(angle: float) -> float:
    """
    Normalize an angle to the range [-pi, pi].
    """

    while angle > math.pi:
        angle -= 2 * math.pi

    while angle < -math.pi:
        angle += 2 * math.pi

    return angle

def distance_to_goal(
    x: float,
    y: float,
    goal_x: float,
    goal_y: float,
) -> float:
    """
    Calculate straight-line distance to the navigation goal.
    """

    dx = goal_x - x
    dy = goal_y - y

    return math.hypot(dx, dy)

def heading_error_to_goal(
    x: float,
    y: float,
    yaw: float,
    goal_x: float,
    goal_y: float,
) -> float:
    """
    Calculate how far the robot must rotate to face the goal.
    """

    dx = goal_x - x
    dy = goal_y - y

    desired_yaw = math.atan2(dy, dx)

    return wrap_angle(
        desired_yaw - yaw
    )
def choose_navigation_action(
    x: float,
    y: float,
    yaw: float,
    goal_x: float,
    goal_y: float,
) -> str:
    """
    Decide whether the robot should ARRIVE, TURN, or MOVE.
    """

    distance = distance_to_goal(
        x,
        y,
        goal_x,
        goal_y,
    )

    # Destination reached
    if distance <= ARRIVAL_DISTANCE_M:
        return "ARRIVED"

    heading_error = heading_error_to_goal(
        x,
        y,
        yaw,
        goal_x,
        goal_y,
    )

    heading_error_deg = math.degrees(
        heading_error
    )

    # Robot is not facing the destination
    if abs(heading_error_deg) > HEADING_TOLERANCE_DEG:
        return "TURN"

    # Robot is approximately facing the destination
    return "MOVE"

def create_navigation_command(
    run_id: str,
    current_x: float,
    current_y: float,
    current_yaw: float,
    destination: str,
    goal: Tuple[float, float],
) -> Tuple[str, RobotCommand | None]:
    """
    Create the next low-level RobotCommand toward a destination.
    """

    goal_x, goal_y = goal

    action = choose_navigation_action(
        current_x,
        current_y,
        current_yaw,
        goal_x,
        goal_y,
    )

    # ----------------------------------
    # ARRIVED
    # ----------------------------------

    if action == "ARRIVED":
        return "ARRIVED", None

    # ----------------------------------
    # Calculate heading
    # ----------------------------------

    heading_error = heading_error_to_goal(
        current_x,
        current_y,
        current_yaw,
        goal_x,
        goal_y,
    )

    # ----------------------------------
    # TURN
    # ----------------------------------

    if action == "TURN":

        wz = (
            TURN_SPEED_RPS
            if heading_error > 0
            else -TURN_SPEED_RPS
        )

        command = RobotCommand(
            run_id=run_id,
            command_id=str(uuid.uuid4()),
            action="TURN",
            vx_mps=0.0,
            vy_mps=0.0,
            wz_rps=wz,
            duration_s=COMMAND_DURATION_S,
            target_waypoint=destination,
            safety_checked=True,
        )

        return "TURN", command

    # ----------------------------------
    # MOVE
    # ----------------------------------

    command = RobotCommand(
        run_id=run_id,
        command_id=str(uuid.uuid4()),
        action="MOVE",
        vx_mps=FORWARD_SPEED_MPS,
        vy_mps=0.0,
        wz_rps=0.0,
        duration_s=COMMAND_DURATION_S,
        target_waypoint=destination,
        safety_checked=True,
    )

    return "MOVE", command

def navigate_from_observation(
    observation: SimulatorObservation,
    destination: str,
):
    """
    Convert a live simulator observation into the next
    navigation command toward the requested destination.
    """

    # Check that the destination exists.
    if destination not in DESTINATION_COORDINATES:
        raise ValueError(
            f"Unsupported destination: {destination}"
        )

    # Get the real ORCA destination coordinate.
    goal_x, goal_y, _ = DESTINATION_COORDINATES[destination]

    # Get the robot's current live pose.
    pose = observation.pose

    return create_navigation_command(
        run_id=observation.run_id,
        current_x=pose.x_m,
        current_y=pose.y_m,
        current_yaw=pose.yaw_rad,
        destination=destination,
        goal=(goal_x, goal_y),
    )
    
def navigate_to_destination(
    adapter,
    destination: str,
    max_steps: int = 100,
):
    """
    Repeatedly observe the robot and send movement commands
    until the destination is reached or max_steps is exceeded.
    """

    for step in range(max_steps):

        # ----------------------------------
        # OBSERVE
        # ----------------------------------

        observation = adapter.get_observation()

        pose = observation.pose

        print(f"\n--- STEP {step} ---")
        print(
            f"Pose: "
            f"x={pose.x_m:.3f}, "
            f"y={pose.y_m:.3f}, "
            f"yaw={math.degrees(pose.yaw_rad):.1f} deg"
        )

        # ----------------------------------
        # DECIDE
        # ----------------------------------

        action, command = navigate_from_observation(
            observation,
            destination,
        )

        print("Decision:", action)

        # ----------------------------------
        # ARRIVED
        # ----------------------------------

        if action == "ARRIVED":

            print()
            print(
                f"ARRIVED at {destination}!"
            )

            return True

        # ----------------------------------
        # ACT
        # ----------------------------------

        if command is not None:
            adapter.send_command(command)

    print()
    print(
        f"Navigation stopped after {max_steps} steps "
        f"without reaching {destination}."
    )

    return False

if __name__ == "__main__":

    adapter = SimAdapter(
        mode="fake"
    )

    try:

        navigate_to_destination(
            adapter=adapter,
            destination="pharmacy",
            max_steps=200,
        )

    finally:

        adapter.close()