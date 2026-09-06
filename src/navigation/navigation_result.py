from typing import Literal

from pydantic import BaseModel


NavigationResultStatus = Literal[
    "ARRIVED",
    "FAILED",
    "MAX_STEPS_EXCEEDED",
    "HUMAN_ASSISTANCE_REQUIRED",
]


class NavigationResult(BaseModel):
    """
    Structured result returned from the navigation system
    to the B2 agent.
    """

    status: NavigationResultStatus

    destination: str

    final_x: float
    final_y: float

    steps_taken: int

    avoidance_count: int = 0

    reason: str