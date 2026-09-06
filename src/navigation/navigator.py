import math
import uuid
from typing import Tuple

from navigation.schemas import RobotCommand, SimulatorObservation
from route_planner import DESTINATION_COORDINATES
from control.adapters.sim_adapter import SimAdapter
from navigation.navigation_state import NavigationState
from navigation.navigation_result import NavigationResult
ARRIVAL_DISTANCE_M = 0.35
HEADING_TOLERANCE_DEG = 12.0

FORWARD_SPEED_MPS = 0.15
TURN_SPEED_RPS = 0.35

COMMAND_DURATION_S = 0.35
OBSTACLE_STOP_M = 0.60
DETOUR_FORWARD_M = 0.80
DETOUR_SIDE_M = 0.80
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

def get_nearest_obstacle_distance(
    observation: SimulatorObservation,
) -> float | None:

    obstacle = get_nearest_obstacle(observation)

    if obstacle is None:
        return None

    return obstacle["distance_m"]

def get_nearest_obstacle(
    observation: SimulatorObservation,
) -> dict | None:
    """
    Return the nearest LiDAR obstacle.

    ORCA currently exposes the nearest sensed surface,
    but using min() keeps this compatible with multiple
    obstacles in the future.
    """

    if not observation.obstacles:
        return None

    valid_obstacles = [
        obstacle
        for obstacle in observation.obstacles
        if "distance_m" in obstacle
    ]

    if not valid_obstacles:
        return None

    return min(
        valid_obstacles,
        key=lambda obstacle: obstacle["distance_m"],
    )

def navigate_from_observation(
    observation: SimulatorObservation,
    destination: str,
):
    """
    Convert a live simulator observation into the next
    safe navigation command.
    """

    if destination not in DESTINATION_COORDINATES:
        raise ValueError(
            f"Unsupported destination: {destination}"
        )

    goal_x, goal_y, _ = DESTINATION_COORDINATES[destination]

    pose = observation.pose

    # -------------------------------------------------
    # 1. ARRIVAL CHECK
    # -------------------------------------------------

    distance = distance_to_goal(
        pose.x_m,
        pose.y_m,
        goal_x,
        goal_y,
    )

    if distance <= ARRIVAL_DISTANCE_M:
        return "ARRIVED", None

    # -------------------------------------------------
    # 2. LIDAR SAFETY CHECK
    # -------------------------------------------------

    nearest_obstacle = get_nearest_obstacle_distance(
        observation
    )

    if (
        nearest_obstacle is not None
        and nearest_obstacle < OBSTACLE_STOP_M
    ):
        command = create_stop_command(
            run_id=observation.run_id,
            destination=destination,
        )

        return "BLOCKED", command

    # -------------------------------------------------
    # 3. NORMAL NAVIGATION
    # -------------------------------------------------

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
    max_steps: int = 300,
):
    """
    Closed-loop navigation with temporary obstacle detours.
    """

    nav_state = create_navigation_state(destination)

    for step in range(max_steps):

        # ==========================================
        # OBSERVE
        # ==========================================

        observation = adapter.get_observation()
        pose = observation.pose

        print(f"\n--- STEP {step} ---")
        print(
            f"Pose: "
            f"x={pose.x_m:.3f}, "
            f"y={pose.y_m:.3f}, "
            f"yaw={math.degrees(pose.yaw_rad):.1f} deg"
        )

        # ==========================================
        # CURRENT ACTIVE GOAL
        # ==========================================

        active_goal = get_active_goal(nav_state)

        if nav_state.avoiding_obstacle:
            print(
                "Mode: AVOIDING OBSTACLE | "
                f"Detour: ({active_goal[0]:.3f}, "
                f"{active_goal[1]:.3f})"
            )
        else:
            print(
                f"Mode: FINAL GOAL | "
                f"Destination: {destination}"
            )

        # ==========================================
        # HAVE WE REACHED THE ACTIVE GOAL?
        # ==========================================

        active_distance = distance_to_goal(
            pose.x_m,
            pose.y_m,
            active_goal[0],
            active_goal[1],
        )

        if active_distance <= ARRIVAL_DISTANCE_M:

            # --------------------------------------
            # Temporary detour reached
            # --------------------------------------

            if nav_state.avoiding_obstacle:

                print("Detour reached.")

                nav_state = finish_obstacle_avoidance(
                    nav_state
                )

                print(
                    f"Resuming final destination: "
                    f"{destination}"
                )

                continue

            # --------------------------------------
            # Final destination reached
            # --------------------------------------

            print()
            print(
                f"ARRIVED at {destination}!"
            )

            return NavigationResult(
                status="ARRIVED",
                destination=destination,

                final_x=pose.x_m,
                final_y=pose.y_m,

                steps_taken=step + 1,

                avoidance_count=nav_state.avoidance_count,

                reason="Destination reached successfully.",
            )

        # ==========================================
        # LIDAR
        # ==========================================

        obstacle = get_nearest_obstacle(
            observation
        )

        # Only start a NEW detour if we're not
        # already following one.
        if (
            not nav_state.avoiding_obstacle
            and obstacle is not None
            and obstacle_requires_avoidance(obstacle)
        ):

            print()
            print("OBSTACLE DETECTED")

            print(
                f"Distance: "
                f"{obstacle['distance_m']:.3f} m"
            )

            side = classify_obstacle_side(
                obstacle
            )

            print("Obstacle side:", side)

            # --------------------------------------
            # STOP FIRST
            # --------------------------------------

            stop_command = create_stop_command(
                run_id=observation.run_id,
                destination=destination,
            )

            adapter.send_command(
                stop_command
            )

            # --------------------------------------
            # CREATE DETOUR
            # --------------------------------------

            nav_state = start_obstacle_avoidance(
                nav_state,
                observation,
                obstacle,
            )
            
            if getattr(adapter, "mode", None) == "fake":
                adapter.set_fake_obstacles([])

            detour = get_active_goal(
                nav_state
            )

            print(
                f"Temporary detour created: "
                f"({detour[0]:.3f}, "
                f"{detour[1]:.3f})"
            )

            continue

        # ==========================================
        # NORMAL MOVEMENT TOWARD ACTIVE GOAL
        # ==========================================

        action, command = navigate_to_goal(
            observation=observation,
            destination=destination,
            goal=active_goal,
        )

        print("Decision:", action)

        if command is not None:
            adapter.send_command(command)

    print()
    print(
        f"Navigation stopped after "
        f"{max_steps} steps."
    )

    observation = adapter.get_observation()
    pose = observation.pose

    return NavigationResult(
        status="MAX_STEPS_EXCEEDED",
        destination=destination,

        final_x=pose.x_m,
        final_y=pose.y_m,

        steps_taken=max_steps,

        avoidance_count=nav_state.avoidance_count,

        reason=(
            "Maximum navigation steps exceeded "
            "before reaching destination."
        ),
    )

        
def create_stop_command(
    run_id: str,
    destination: str,
) -> RobotCommand:
    """
    Create an immediate safe stop command.
    """

    return RobotCommand(
        run_id=run_id,
        command_id=str(uuid.uuid4()),
        action="STOP",
        vx_mps=0.0,
        vy_mps=0.0,
        wz_rps=0.0,
        duration_s=0.25,
        target_waypoint=destination,
        safety_checked=True,
    )
    

        
def classify_obstacle_side(
    obstacle: dict,
) -> str:
    """
    Classify the obstacle relative to the robot.

    ORCA convention:
    +x = forward
    +y = left
    -y = right
    """

    relative_x = obstacle.get("relative_x_m", 0.0)
    relative_y = obstacle.get("relative_y_m", 0.0)

    # Obstacle is behind the robot.
    if relative_x < 0:
        return "BEHIND"

    # Approximately straight ahead.
    if abs(relative_y) < 0.25:
        return "FRONT"

    if relative_y > 0:
        return "LEFT"

    return "RIGHT"

def obstacle_requires_avoidance(
    obstacle: dict,
) -> bool:
    """
    Return True when an obstacle is close enough and
    positioned where it may interfere with navigation.
    """

    distance = obstacle.get("distance_m")

    if distance is None:
        return False

    if distance >= OBSTACLE_STOP_M:
        return False

    relative_x = obstacle.get("relative_x_m", 0.0)

    # Ignore obstacles behind the robot.
    if relative_x <= 0:
        return False

    return True

def create_detour_goal(
    observation: SimulatorObservation,
    obstacle: dict,
) -> tuple[float, float]:
    """
    Create a temporary world-coordinate waypoint around
    the detected obstacle.

    This is a lightweight reactive avoidance strategy for
    the MVP, not a global path planner.
    """

    pose = observation.pose

    obstacle_side = classify_obstacle_side(obstacle)

    # ----------------------------------------
    # Choose the opposite side.
    # ----------------------------------------

    if obstacle_side == "LEFT":
        lateral_offset = -DETOUR_SIDE_M

    elif obstacle_side == "RIGHT":
        lateral_offset = DETOUR_SIDE_M

    else:
        # Obstacle directly ahead.
        #
        # For the first MVP choose the left side.
        # Later camera / free-space analysis can choose
        # whichever side is actually safer.
        lateral_offset = DETOUR_SIDE_M

    forward_offset = DETOUR_FORWARD_M

    # ----------------------------------------
    # Convert robot-relative detour coordinates
    # into ORCA world coordinates.
    # ----------------------------------------

    c = math.cos(pose.yaw_rad)
    s = math.sin(pose.yaw_rad)

    world_dx = (
        c * forward_offset
        - s * lateral_offset
    )

    world_dy = (
        s * forward_offset
        + c * lateral_offset
    )

    detour_x = pose.x_m + world_dx
    detour_y = pose.y_m + world_dy

    return detour_x, detour_y



def create_navigation_state(
    destination: str,
) -> NavigationState:
    """
    Create local navigation state for a hospital destination.
    """

    if destination not in DESTINATION_COORDINATES:
        raise ValueError(
            f"Unsupported destination: {destination}"
        )

    goal_x, goal_y, _ = DESTINATION_COORDINATES[destination]

    return NavigationState(
        destination=destination,
        goal_x=goal_x,
        goal_y=goal_y,
    )
    
def get_active_goal(
    state: NavigationState,
) -> tuple[float, float]:
    """
    Return the coordinate the robot should currently pursue.

    During obstacle avoidance this is the temporary detour.
    Otherwise it is the final destination.
    """

    if (
        state.avoiding_obstacle
        and state.detour_x is not None
        and state.detour_y is not None
    ):
        return state.detour_x, state.detour_y

    return state.goal_x, state.goal_y

def start_obstacle_avoidance(
    state: NavigationState,
    observation: SimulatorObservation,
    obstacle: dict,
) -> NavigationState:
    """
    Create and activate a temporary detour goal.
    """

    detour_x, detour_y = create_detour_goal(
        observation,
        obstacle,
    )

    state.detour_x = detour_x
    state.detour_y = detour_y

    state.avoiding_obstacle = True
    state.avoidance_count += 1

    return state

def finish_obstacle_avoidance(
    state: NavigationState,
) -> NavigationState:
    """
    Clear the temporary detour and resume the final goal.
    """

    state.detour_x = None
    state.detour_y = None
    state.avoiding_obstacle = False

    return state


        
def navigate_to_goal(
    observation: SimulatorObservation,
    destination: str,
    goal: tuple[float, float],
):
    """
    Generate the next command toward any coordinate goal.

    The goal may be either:
    - the final hospital destination, or
    - a temporary obstacle-avoidance detour.
    """

    goal_x, goal_y = goal
    pose = observation.pose

    distance = distance_to_goal(
        pose.x_m,
        pose.y_m,
        goal_x,
        goal_y,
    )

    if distance <= ARRIVAL_DISTANCE_M:
        return "ARRIVED", None

    return create_navigation_command(
        run_id=observation.run_id,
        current_x=pose.x_m,
        current_y=pose.y_m,
        current_yaw=pose.yaw_rad,
        destination=destination,
        goal=(goal_x, goal_y),
    )
    
if __name__ == "__main__":

    adapter = SimAdapter(mode="fake")

    adapter.set_fake_obstacles([
        {
            "id": "fake_obstacle",
            "distance_m": 0.40,

            "relative_x_m": 0.40,
            "relative_y_m": 0.0,

            "world_x_m": 0.40,
            "world_y_m": 0.0,
            "world_z_m": 0.5,
        }
    ])

    try:
        result = navigate_to_destination(
            adapter=adapter,
            destination="pharmacy",
            max_steps=300,
        )

        print("\n--- NAVIGATION RESULT ---")
        print("Status:", result.status)
        print("Destination:", result.destination)
        print(
            "Final position:",
            (result.final_x, result.final_y),
        )
        print("Steps:", result.steps_taken)
        print(
            "Avoidance count:",
            result.avoidance_count,
        )
        print("Reason:", result.reason)

    finally:
        adapter.close()