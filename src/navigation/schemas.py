from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# CONTRACT A: Agent Architecture Lead -> Navigation Lead
# ---------------------------------------------------------

DestinationID = Literal[
    "radiology", 
    "ward_5K", 
    "pharmacy", 
    "consultation_room", 
    "elevator", 
    "restroom", 
    "reception"
]

AgentAction = Literal[
    "MOVE_TO_WAYPOINT", 
    "TURN_LEFT", 
    "TURN_RIGHT", 
    "YIELD", 
    "STOP", 
    "REPLAN", 
    "ARRIVED", 
    "ASK_CLARIFICATION"
]

class AgentNavigationCommand(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    command_id: str
    destination: DestinationID
    route_id: str
    waypoint_id: str
    waypoint_index: int = Field(ge=0)
    total_waypoints: int = Field(gt=0)
    action: AgentAction
    max_duration_s: float = Field(gt=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    announcement: Optional[str] = None

# ---------------------------------------------------------
# CONTRACT B: Navigation Lead -> Simulation Lead
# ---------------------------------------------------------

RobotAction = Literal["MOVE", "TURN", "STOP", "HOLD"]

class RobotCommand(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    command_id: str
    action: RobotAction
    vx_mps: float
    vy_mps: float
    wz_rps: float
    duration_s: float = Field(ge=0.0)
    target_waypoint: str
    safety_checked: bool

# ---------------------------------------------------------
# CONTRACT C: Simulation Lead -> Navigation Lead
# ---------------------------------------------------------

RobotStatus = Literal["RUNNING", "REACHED", "STOPPED", "COLLISION", "ERROR"]

class Pose(BaseModel):
    x_m: float
    y_m: float
    yaw_rad: float

class Velocity(BaseModel):
    vx_mps: float
    vy_mps: float
    wz_rps: float

class SimulatorObservation(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    command_id: str
    timestamp_ms: int
    sim_time_s: float
    robot_status: RobotStatus
    pose: Pose
    velocity: Velocity
    current_waypoint: Optional[str] = None
    target_waypoint: Optional[str] = None
    distance_to_target_m: Optional[float] = None
    heading_error_rad: float
    camera_frame: Optional[str] = None
    # Flexible dict to accommodate slightly different obstacle schemas in the PDFs
    obstacles: List[Dict[str, Any]] = Field(default_factory=list)

# ---------------------------------------------------------
# CONTRACT D: Navigation Lead -> Agent Architecture Lead
# ---------------------------------------------------------

FeedbackStatus = Literal[
    "EXECUTED", 
    "WAYPOINT_REACHED", 
    "BLOCKED", 
    "YIELDED", 
    "STOPPED", 
    "REPLAN_REQUIRED", 
    "ARRIVED", 
    "INVALID_COMMAND", 
    "ERROR", 
    "HUMAN_ASSISTANCE_REQUIRED"
]

class NavigationFeedback(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    command_id: str
    status: FeedbackStatus
    current_waypoint: str
    target_waypoint: str
    distance_to_target_m: Optional[float] = None
    obstacle_detected: bool
    obstacle_distance_m: Optional[float] = None
    replan_required: bool
    human_assistance_required: bool
    reason: str
    recommended_next_action: Optional[str] = None