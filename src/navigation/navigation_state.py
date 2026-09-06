from typing import Optional

from pydantic import BaseModel


class NavigationState(BaseModel):
    """
    Runtime state for closed-loop local navigation.
    """

    # Final hospital destination.
    destination: str

    # Final destination coordinate.
    goal_x: float
    goal_y: float

    # Temporary local detour target.
    detour_x: Optional[float] = None
    detour_y: Optional[float] = None

    # Whether the robot is currently following a detour.
    avoiding_obstacle: bool = False

    # Number of avoidance attempts.
    avoidance_count: int = 0