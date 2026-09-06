"""
Route planner for the hospital robot guide.

Current MVP:
- Uses real ORCA destination coordinates.
- Uses temporary routing waypoints / graph until the full ORCA
  walkable map is available.
- Supports initial route planning.
- Supports replanning around blocked graph locations.

Future:
- Replace the temporary graph with coordinate-based A*.
- Use the real hospital boundaries / occupancy map.
- Convert LiDAR obstacles into temporary blocked regions.
"""

import uuid

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel


# ============================================================
# ROUTE DATA MODELS
# ============================================================


class RouteWaypoint(BaseModel):
    """
    One physical waypoint in the ORCA world.
    """

    waypoint_id: str

    x_m: float
    y_m: float


class RoutePlan(BaseModel):
    """
    Complete route returned to the B2 agent.
    """

    route_id: str

    destination: str

    waypoints: List[RouteWaypoint]


# ============================================================
# REAL ORCA DESTINATION COORDINATES
# ============================================================
#
# These are the actual destination coordinates from the
# hospital ORCA environment.
#
# Format:
# destination -> (x, y, z)
#
# The current route planner only needs x and y.
# ============================================================


DESTINATION_COORDINATES: Dict[
    str,
    Tuple[float, float, float],
] = {

    "radiology": (
        3.305,
        5.411,
        0.0,
    ),

    "ward_5K": (
        5.181,
        0.218,
        0.0,
    ),

    "pharmacy": (
        -4.257,
        -3.314,
        0.0,
    ),

    "consultation_room": (
        1.232,
        -5.791,
        0.0,
    ),

    "elevator": (
        1.791,
        -7.711,
        0.0,
    ),

    "restroom": (
        -4.691,
        2.201,
        0.0,
    ),

    "reception": (
        -1.337,
        4.941,
        0.0,
    ),
}


# ============================================================
# REAL ORCA BOUNDARY INFORMATION
# ============================================================
#
# These values come from the ORCA hospital environment.
#
# IMPORTANT:
#
# Some boundary definitions are currently incomplete.
# Therefore they are stored here for future A* integration,
# but they are NOT yet treated as complete obstacle rectangles.
#
# Once the exact hospital geometry is finalized, these values
# can be converted into an occupancy grid.
# ============================================================


ENVIRONMENT_BOUNDARIES = {

    "radiology": {
        "reference_x": 1.008,
        "reference_y": 2.956,
        "boundary_y": 0.664,
    },

    "ward_5K": {
        "y_max": 0.670,
        "y_min": -1.292,
    },

    "consultation_room": {
        "x_max": 1.602,
        "x_min": -0.075,
    },

    "elevator": {
        "x_max": 1.602,
        "x_min": -0.075,
        "y_max": -8.521,
    },

    "pharmacy": {
        "x_max": -2.060,
        "x_min": -3.171,
        "y_max": -1.296,
        "y_min": -2.808,
    },

    "restroom": {
        "y_max": 2.258,
        "y_min": 1.075,
    },

    "reception": {
        "x_max": 1.008,
        "x_min": -2.141,
    },
}


# ============================================================
# TEMPORARY ROUTING WAYPOINTS
# ============================================================
#
# These are NOT real ORCA corridor coordinates.
#
# They currently exist only so that:
#
# Agent
#   -> route planner
#   -> waypoint commands
#   -> feedback
#   -> replanning
#
# can be developed before the complete ORCA navigation map
# is available.
#
# Replace these with real map waypoints / A* later.
# ============================================================


TEMPORARY_ROUTING_WAYPOINTS: Dict[str, RouteWaypoint] = {

    "lobby": RouteWaypoint(
        waypoint_id="lobby",
        x_m=0.0,
        y_m=0.0,
    ),

    "corridor_a": RouteWaypoint(
        waypoint_id="corridor_a",
        x_m=3.0,
        y_m=0.0,
    ),

    "corridor_b": RouteWaypoint(
        waypoint_id="corridor_b",
        x_m=6.0,
        y_m=0.0,
    ),

    "corridor_c": RouteWaypoint(
        waypoint_id="corridor_c",
        x_m=9.0,
        y_m=2.0,
    ),

    "corridor_d": RouteWaypoint(
        waypoint_id="corridor_d",
        x_m=6.0,
        y_m=4.0,
    ),
}


# ============================================================
# COMBINED LOCATION LOOKUP
# ============================================================


LOCATIONS: Dict[str, RouteWaypoint] = {
    **TEMPORARY_ROUTING_WAYPOINTS,
}


# Add the REAL destination coordinates.


for destination_id, coordinates in DESTINATION_COORDINATES.items():

    x_m, y_m, _ = coordinates

    LOCATIONS[destination_id] = RouteWaypoint(
        waypoint_id=destination_id,
        x_m=x_m,
        y_m=y_m,
    )


# ============================================================
# TEMPORARY HOSPITAL GRAPH
# ============================================================
#
# This graph is used only during the MVP development stage.
#
# It will eventually be replaced by:
#
#     ORCA map
#         ↓
#     occupancy grid
#         ↓
#        A*
#
# ============================================================


TEMPORARY_GRAPH: Dict[str, List[str]] = {

    "lobby": [
        "corridor_a",
        "reception",
    ],

    "reception": [
        "lobby",
    ],

    "corridor_a": [
        "lobby",
        "corridor_b",
        "corridor_d",
        "elevator",
        "ward_5K",
    ],

    "corridor_b": [
        "corridor_a",
        "corridor_c",
        "corridor_d",
    ],

    "corridor_c": [
        "corridor_b",
        "corridor_d",
        "radiology",
        "restroom",
    ],

    "corridor_d": [
        "corridor_a",
        "corridor_b",
        "corridor_c",
        "pharmacy",
    ],

    "radiology": [
        "corridor_c",
    ],

    "ward_5K": [
        "corridor_a",
    ],

    "pharmacy": [
        "corridor_d",
        "consultation_room",
    ],

    "consultation_room": [
        "pharmacy",
    ],

    "restroom": [
        "corridor_c",
    ],

    "elevator": [
        "corridor_a",
    ],
}


# ============================================================
# DESTINATION HELPERS
# ============================================================


def is_supported_destination(
    destination: str,
) -> bool:
    """
    Return True when the destination exists in the
    real ORCA destination data.
    """

    return destination in DESTINATION_COORDINATES


def get_destination_waypoint(
    destination: str,
) -> Optional[RouteWaypoint]:
    """
    Return the real ORCA coordinate for a destination.
    """

    if not is_supported_destination(destination):
        return None

    return LOCATIONS[destination]


# ============================================================
# PATH SEARCH
# ============================================================


def find_path(
    start: str,
    destination: str,
    blocked_locations: Optional[Set[str]] = None,
) -> Optional[List[str]]:
    """
    Find a path through the temporary hospital graph.

    BFS is appropriate for the current unweighted MVP graph.

    blocked_locations contains locations that must not be used.

    Example:

        blocked_locations = {"corridor_b"}

    The route planner will then attempt to find another route.
    """

    if blocked_locations is None:
        blocked_locations = set()

    # ----------------------------------------
    # Validate start
    # ----------------------------------------

    if start not in TEMPORARY_GRAPH:
        return None

    # ----------------------------------------
    # Validate destination
    # ----------------------------------------

    if destination not in TEMPORARY_GRAPH:
        return None

    # ----------------------------------------
    # Cannot start or finish inside
    # a blocked location
    # ----------------------------------------

    if start in blocked_locations:
        return None

    if destination in blocked_locations:
        return None

    # ----------------------------------------
    # BFS queue
    #
    # Each entry contains:
    #
    # current location
    # path taken to reach it
    # ----------------------------------------

    queue = deque([
        (
            start,
            [start],
        )
    ])

    visited = {start}

    # ----------------------------------------
    # Search
    # ----------------------------------------

    while queue:

        current, path = queue.popleft()

        if current == destination:
            return path

        for neighbor in TEMPORARY_GRAPH.get(
            current,
            [],
        ):

            if neighbor in visited:
                continue

            if neighbor in blocked_locations:
                continue

            visited.add(neighbor)

            queue.append(
                (
                    neighbor,
                    path + [neighbor],
                )
            )

    # No safe path found.

    return None


# ============================================================
# INITIAL ROUTE PLANNING
# ============================================================


def plan_route(
    start: str,
    destination: str,
    blocked_locations: Optional[Set[str]] = None,
) -> Optional[RoutePlan]:
    """
    Generate an initial route from the robot's current
    location to the visitor's destination.

    The destination coordinates are REAL ORCA coordinates.

    Intermediate corridor waypoints are temporary until
    the full hospital navigation map is available.
    """

    # ----------------------------------------
    # Destination must exist in ORCA
    # ----------------------------------------

    if not is_supported_destination(destination):
        return None

    # ----------------------------------------
    # Search for path
    # ----------------------------------------

    path = find_path(
        start=start,
        destination=destination,
        blocked_locations=blocked_locations,
    )

    if path is None:
        return None

    # ----------------------------------------
    # Remove the starting position.
    #
    # The robot is already there.
    #
    # Example:
    #
    # lobby
    # corridor_a
    # corridor_b
    # radiology
    #
    # becomes:
    #
    # corridor_a
    # corridor_b
    # radiology
    # ----------------------------------------

    remaining_path = path[1:]

    # ----------------------------------------
    # Convert IDs into physical waypoints
    # ----------------------------------------

    waypoints = [
        LOCATIONS[location]
        for location in remaining_path
    ]

    # ----------------------------------------
    # Create route
    # ----------------------------------------

    return RoutePlan(
        route_id=str(uuid.uuid4()),
        destination=destination,
        waypoints=waypoints,
    )


# ============================================================
# REPLANNING
# ============================================================


def replan_route(
    current_location: str,
    destination: str,
    blocked_locations: Set[str],
) -> Optional[RoutePlan]:
    """
    Generate a new route after the current route becomes
    unavailable.

    The original destination is preserved.

    Example:

        Goal:
            radiology

        Original route:
            corridor_a
            corridor_b
            corridor_c
            radiology

        corridor_b becomes blocked.

        Replanned route:
            corridor_d
            corridor_c
            radiology
    """

    return plan_route(
        start=current_location,
        destination=destination,
        blocked_locations=blocked_locations,
    )