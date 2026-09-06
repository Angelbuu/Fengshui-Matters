"""Build a 2-D navigation map from the actual Orca hospital scene."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


# ------------------------------------------------------------
# Known destination locations
# ------------------------------------------------------------

DESTINATIONS = {
    "radiology": (3.305, 5.411),
    "ward_5K": (5.181, 0.218),
    "pharmacy": (-4.257, -3.314),
    "consultation_room": (1.232, -5.791),
    "elevator": (1.791, -7.711),
    "restroom": (-4.691, 2.201),
    "reception": (-1.337, 4.941),
}


# ------------------------------------------------------------
# Map configuration
# ------------------------------------------------------------

# Covers your current hospital coordinates with some margin.
MAP_X_MIN = -6.0
MAP_X_MAX = 6.5

MAP_Y_MIN = -9.5
MAP_Y_MAX = 7.0

# 10 cm cells.
GRID_RESOLUTION_M = 0.10

# Keep the Go2 centre this far away from obstacles.
ROBOT_CLEARANCE_M = 0.35


@dataclass
class OccupancyMap:
    grid: np.ndarray

    x_min: float
    y_min: float

    resolution: float

    def world_to_grid(
        self,
        x: float,
        y: float,
    ) -> tuple[int, int]:

        col = int(
            round(
                (x - self.x_min)
                / self.resolution
            )
        )

        row = int(
            round(
                (y - self.y_min)
                / self.resolution
            )
        )

        return row, col

    def nearest_free_cell(
        self,
        x: float,
        y: float,
        max_radius_m: float = 1.0,
    ) -> tuple[int, int] | None:
        """
        Return the nearest unblocked grid cell to a world coordinate.

        Destination markers may sit beside walls or inside the inflated
        safety margin, so the robot should navigate to a safe approach
        point rather than the exact marker centre.
        """

        start_row, start_col = self.world_to_grid(x, y)

        rows, cols = self.grid.shape

        max_cells = int(
            math.ceil(max_radius_m / self.resolution)
        )

        best = None
        best_distance = float("inf")

        for dr in range(-max_cells, max_cells + 1):
            for dc in range(-max_cells, max_cells + 1):

                row = start_row + dr
                col = start_col + dc

                if not (
                    0 <= row < rows
                    and 0 <= col < cols
                ):
                    continue

                if self.grid[row, col]:
                    continue

                wx, wy = self.grid_to_world(row, col)

                distance = math.hypot(
                    wx - x,
                    wy - y,
                )

                if distance < best_distance:
                    best_distance = distance
                    best = (row, col)

        return best

    def grid_to_world(
        self,
        row: int,
        col: int,
    ) -> tuple[float, float]:

        x = (
            self.x_min
            + col * self.resolution
        )

        y = (
            self.y_min
            + row * self.resolution
        )

        return x, y


def _is_partition(name: str) -> bool:
    """
    Your partition geoms currently have generated names rather
    than a useful 'partition' prefix.

    They are tall, thin box geoms, so geometry classification
    below is used in addition to this name check.
    """

    lowered = name.lower()

    return (
        "partition" in lowered
        or "wall" in lowered
    )


def _is_box(name: str) -> bool:

    lowered = name.lower()

    return (
        "corrugated_cardboard_box" in lowered
        or "cardboard_box" in lowered
    )


def _is_destination_marker(
    name: str,
) -> bool:

    lowered = name.lower()

    return (
        "blue_flammable_liquid_drums" in lowered
        or "cabinet" in lowered
    )


def _looks_like_partition(
    geom: dict,
) -> bool:
    """
    Detect your generated partition collision boxes.

    From the Orca dump they are approximately:
        size=[0.026, 0.65, 1.0]

    MuJoCo box sizes are half-extents.
    """

    if geom["type"] != "box":
        return False

    size = geom["size"]

    if len(size) < 3:
        return False

    sx, sy, sz = size[:3]

    return (
        sz >= 0.7
        and min(sx, sy) <= 0.08
        and max(sx, sy) >= 0.35
    )


def should_block(
    geom: dict,
) -> bool:

    name = geom["name"]

    # Never make destination markers obstacles.
    if _is_destination_marker(name):
        return False

    # Ignore robot geometry.
    if (
        "quadruped_robot" in name.lower()
        or "go2_" in name.lower()
    ):
        return False

    # Ignore the huge ground plane.
    if geom["type"] == "plane":
        return False

    # Boxes are obstacles.
    if _is_box(name):
        return True

    # Partitions/walls are obstacles.
    if (
        _is_partition(name)
        or _looks_like_partition(geom)
    ):
        return True

    return False


def build_occupancy_map(
    model,
    agent_names,
) -> OccupancyMap:
    """
    Convert Orca MuJoCo collision geometry into a 2-D occupancy grid.
    """

    import mujoco

    rows = int(
        math.ceil(
            (MAP_Y_MAX - MAP_Y_MIN)
            / GRID_RESOLUTION_M
        )
    ) + 1

    cols = int(
        math.ceil(
            (MAP_X_MAX - MAP_X_MIN)
            / GRID_RESOLUTION_M
        )
    ) + 1

    grid = np.zeros(
        (rows, cols),
        dtype=np.uint8,
    )

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    agent_prefixes = tuple(
        f"{name}_"
        for name in agent_names
    )

    blocked_count = 0

    for geom_id in range(model.ngeom):

        body_id = int(
            model.geom_bodyid[geom_id]
        )

        body_name = (
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
            )
            or ""
        )

        geom_name = (
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_id,
            )
            or f"geom_{geom_id}"
        )

        # Exclude runtime robot.
        if (
            body_name.startswith(agent_prefixes)
            or geom_name.startswith(agent_prefixes)
        ):
            continue

        geom_type = (
            mujoco.mjtGeom(
                int(model.geom_type[geom_id])
            )
            .name
            .removeprefix("mjGEOM_")
            .lower()
        )

        geom = {
            "name": geom_name,
            "body": body_name,
            "type": geom_type,
            "size": (
                model.geom_size[geom_id]
                .tolist()
            ),
        }

        if not should_block(geom):
            continue

        blocked_count += 1

        world_pos = data.geom_xpos[
            geom_id
        ]

        cx = float(world_pos[0])
        cy = float(world_pos[1])

        # World rotation matrix for this geom.
        rotation = (
            data.geom_xmat[geom_id]
            .reshape(3, 3)
        )

        size = model.geom_size[
            geom_id
        ]

        # ----------------------------------------------------
        # ----------------------------------------------------
        # Rasterize the actual oriented obstacle footprint.
        #
        # Previously we filled the entire axis-aligned bounding
        # box of a rotated partition. That greatly overestimated
        # angled walls and could close visually open corridors.
        #
        # Here we:
        #   1. find a conservative search box,
        #   2. transform each candidate grid cell into the geom's
        #      local XY frame,
        #   3. block only cells inside the real oriented footprint
        #      plus Go2 clearance.
        # ----------------------------------------------------

        if geom_type == "box":

            hx = float(size[0])
            hy = float(size[1])

            # Conservative world-space search bounds only.
            search_hx = (
                abs(rotation[0, 0]) * hx
                + abs(rotation[0, 1]) * hy
                + ROBOT_CLEARANCE_M
            )

            search_hy = (
                abs(rotation[1, 0]) * hx
                + abs(rotation[1, 1]) * hy
                + ROBOT_CLEARANCE_M
            )

        elif geom_type in {
            "cylinder",
            "sphere",
        }:

            radius = (
                float(size[0])
                + ROBOT_CLEARANCE_M
            )

            search_hx = radius
            search_hy = radius

        else:
            continue

        x0 = cx - search_hx
        x1 = cx + search_hx

        y0 = cy - search_hy
        y1 = cy + search_hy

        col0 = max(
            0,
            int(
                math.floor(
                    (x0 - MAP_X_MIN)
                    / GRID_RESOLUTION_M
                )
            ),
        )

        col1 = min(
            cols - 1,
            int(
                math.ceil(
                    (x1 - MAP_X_MIN)
                    / GRID_RESOLUTION_M
                )
            ),
        )

        row0 = max(
            0,
            int(
                math.floor(
                    (y0 - MAP_Y_MIN)
                    / GRID_RESOLUTION_M
                )
            ),
        )

        row1 = min(
            rows - 1,
            int(
                math.ceil(
                    (y1 - MAP_Y_MIN)
                    / GRID_RESOLUTION_M
                )
            ),
        )

        # Rotation columns are the geom's local axes expressed
        # in world coordinates.
        axis_x_x = float(rotation[0, 0])
        axis_x_y = float(rotation[1, 0])

        axis_y_x = float(rotation[0, 1])
        axis_y_y = float(rotation[1, 1])

        for row in range(
            row0,
            row1 + 1,
        ):
            for col in range(
                col0,
                col1 + 1,
            ):

                wx = (
                    MAP_X_MIN
                    + col * GRID_RESOLUTION_M
                )

                wy = (
                    MAP_Y_MIN
                    + row * GRID_RESOLUTION_M
                )

                dx = wx - cx
                dy = wy - cy

                if geom_type == "box":

                    # World displacement -> geom-local XY.
                    local_x = (
                        dx * axis_x_x
                        + dy * axis_x_y
                    )

                    local_y = (
                        dx * axis_y_x
                        + dy * axis_y_y
                    )

                    blocked = (
                        abs(local_x)
                        <= hx + ROBOT_CLEARANCE_M
                        and
                        abs(local_y)
                        <= hy + ROBOT_CLEARANCE_M
                    )

                else:

                    # Circular obstacle footprint.
                    blocked = (
                        dx * dx + dy * dy
                        <= radius * radius
                    )

                if blocked:
                    grid[row, col] = 1

    print(
        f"[OrcaMap] blocked geoms: "
        f"{blocked_count}"
    )

    print(
        f"[OrcaMap] grid: "
        f"{cols} x {rows}"
    )

    return OccupancyMap(
        grid=grid,
        x_min=MAP_X_MIN,
        y_min=MAP_Y_MIN,
        resolution=GRID_RESOLUTION_M,
    )


# ============================================================
# A* PATH PLANNING
# ============================================================

def astar_path(
    nav_map: OccupancyMap,
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
) -> list[tuple[float, float]] | None:
    """
    Find a collision-free 2-D path through the Orca occupancy grid.

    Returns world-coordinate path points.
    """

    import heapq

    start = nav_map.nearest_free_cell(
        start_x,
        start_y,
        max_radius_m=0.8,
    )

    goal = nav_map.nearest_free_cell(
        goal_x,
        goal_y,
        max_radius_m=1.0,
    )

    if start is None:
        print("[A*] No free start cell.")
        return None

    if goal is None:
        print("[A*] No free goal cell.")
        return None

    rows, cols = nav_map.grid.shape

    # 8-connected grid.
    neighbors = [
        (-1,  0, 1.0),
        ( 1,  0, 1.0),
        ( 0, -1, 1.0),
        ( 0,  1, 1.0),

        (-1, -1, math.sqrt(2.0)),
        (-1,  1, math.sqrt(2.0)),
        ( 1, -1, math.sqrt(2.0)),
        ( 1,  1, math.sqrt(2.0)),
    ]

    def heuristic(a, b):
        return math.hypot(
            a[0] - b[0],
            a[1] - b[1],
        )

    open_heap = []

    heapq.heappush(
        open_heap,
        (
            heuristic(start, goal),
            0.0,
            start,
        ),
    )

    came_from = {}

    g_score = {
        start: 0.0,
    }

    visited = set()

    while open_heap:

        _, current_cost, current = (
            heapq.heappop(open_heap)
        )

        if current in visited:
            continue

        visited.add(current)

        if current == goal:
            break

        row, col = current

        for dr, dc, move_cost in neighbors:

            nr = row + dr
            nc = col + dc

            if not (
                0 <= nr < rows
                and 0 <= nc < cols
            ):
                continue

            if nav_map.grid[nr, nc]:
                continue

            # Prevent diagonal corner-cutting.
            if dr != 0 and dc != 0:
                if (
                    nav_map.grid[row + dr, col]
                    or nav_map.grid[row, col + dc]
                ):
                    continue

            neighbor = (nr, nc)

            tentative = (
                current_cost + move_cost
            )

            if tentative >= g_score.get(
                neighbor,
                float("inf"),
            ):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative

            priority = (
                tentative
                + heuristic(neighbor, goal)
            )

            heapq.heappush(
                open_heap,
                (
                    priority,
                    tentative,
                    neighbor,
                ),
            )

    if goal not in g_score:
        print("[A*] No path found.")
        return None

    # Reconstruct grid path.
    cells = [goal]

    current = goal

    while current != start:
        current = came_from[current]
        cells.append(current)

    cells.reverse()

    return [
        nav_map.grid_to_world(row, col)
        for row, col in cells
    ]


def simplify_path(
    path: list[tuple[float, float]],
    spacing_m: float = 0.50,
) -> list[tuple[float, float]]:
    """
    Reduce dense 10-cm A* cells into manageable Go2 waypoints.

    Always preserves the final path point.
    """

    if len(path) <= 2:
        return path

    result = [
        path[0],
    ]

    last_x, last_y = path[0]

    for x, y in path[1:-1]:

        distance = math.hypot(
            x - last_x,
            y - last_y,
        )

        if distance >= spacing_m:

            result.append(
                (x, y)
            )

            last_x = x
            last_y = y

    if result[-1] != path[-1]:
        result.append(path[-1])

    return result
