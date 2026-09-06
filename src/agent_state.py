from typing import List, Optional, Set
from pydantic import BaseModel
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """
    Stores the current state of one hospital
    navigation session.
    """

    # One ID for the entire visitor guidance session
    run_id: str

    # Final destination selected by B1
    destination: Optional[str] = None

    # Whether the destination is a department or ward
    destination_type: Optional[str] = None

    # Ward number when destination_type == WARD
    ward_number: Optional[int] = None

    # Confidence produced by B1
    confidence: float = 0.0

    # Current navigation route
    route_id: Optional[str] = None
    # Current route assigned by the route planner
    current_route: List[str] = Field(default_factory=list)
    blocked_locations: Set[str] = Field(default_factory=set)

    # Robot's current position in the route
    current_waypoint: Optional[str] = None

    waypoint_index: int = 0
    total_waypoints: int = 0

    # Current agent status
    status: str = "IDLE"

    # Navigation/environment flags
    replan_required: bool = False
    human_assistance_required: bool = False