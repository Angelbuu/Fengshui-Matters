import uuid
from llm_agent_destination import (
    DestinationDecision,
    Intent,
    build_resolver,
)

from agent_state import AgentState
from navigation.command_validator import validate_agent_command
from navigation.schemas import NavigationFeedback
from navigation.route_planner import RoutePlan, plan_route, replan_route

def handle_destination_decision(
    decision: DestinationDecision,
) -> AgentState:
    """
    Convert B1's destination decision into
    the initial B2 agent state.
    """

    run_id = str(uuid.uuid4())

    # ----------------------------------------
    # NAVIGATE
    # ----------------------------------------

    if decision.intent == Intent.NAVIGATE:

        return AgentState(
            run_id=run_id,
            destination=decision.destination,
            confidence=decision.confidence,
            status="READY_TO_NAVIGATE",
        )

    # ----------------------------------------
    # CLARIFY
    # ----------------------------------------

    if decision.intent == Intent.CLARIFY:

        return AgentState(
            run_id=run_id,
            confidence=decision.confidence,
            status="WAITING_FOR_CLARIFICATION",
        )

    # ----------------------------------------
    # UNKNOWN
    # ----------------------------------------

    return AgentState(
        run_id=run_id,
        confidence=decision.confidence,
        status="UNKNOWN_DESTINATION",
    )
    
def create_navigation_command(
    state: AgentState,
    route_id: str,
    waypoint_id: str,
    waypoint_index: int,
    total_waypoints: int,
):
    """
    Create a navigation command from the current agent state.
    """

    raw_command = {
        "run_id": state.run_id,
        "command_id": str(uuid.uuid4()),

        "destination": state.destination,

        "route_id": route_id,
        "waypoint_id": waypoint_id,
        "waypoint_index": waypoint_index,
        "total_waypoints": total_waypoints,

        "action": "MOVE_TO_WAYPOINT",

        "max_duration_s": 10.0,

        "confidence": state.confidence,

        "announcement": None,
    }

    return validate_agent_command(raw_command)

    
def handle_navigation_feedback(
    state: AgentState,
    feedback: NavigationFeedback,
) -> AgentState:
    """
    Update the agent state based on feedback from navigation.
    """

    # Always update our latest known position
    state.current_waypoint = feedback.current_waypoint

    # Human intervention has the highest priority
    if feedback.human_assistance_required:
        state.status = "NEEDS_HUMAN_ASSISTANCE"
        state.human_assistance_required = True
        return state

    # Navigation cannot continue using the current route
    if (
        feedback.replan_required
        or feedback.status == "REPLAN_REQUIRED"
        or feedback.status == "BLOCKED"
    ):
        state.status = "REPLANNING"
        state.replan_required = True
        return state

    # Destination successfully reached
    if feedback.status == "ARRIVED":
        state.status = "ARRIVED"
        state.replan_required = False
        return state

    # Current waypoint successfully reached
    if feedback.status == "WAYPOINT_REACHED":

        state.waypoint_index += 1

        if state.waypoint_index >= state.total_waypoints:
            state.status = "ARRIVED"
        else:
            state.status = "READY_FOR_NEXT_WAYPOINT"

        return state

    # Robot successfully executed the current command
    if feedback.status == "EXECUTED":
        state.status = "NAVIGATING"
        return state

    # Robot yielded to another person / obstacle
    if feedback.status == "YIELDED":
        state.status = "YIELDED"
        return state

    # Navigation stopped
    if feedback.status == "STOPPED":
        state.status = "STOPPED"
        return state

    # Something went wrong
    if feedback.status in ("ERROR", "INVALID_COMMAND"):
        state.status = "ERROR"
        return state

    return state

def decide_next_action(state: AgentState) -> str:
    """
    Decide what the agent should do next based on its current state.
    """

    if state.human_assistance_required:
        return "REQUEST_HUMAN_ASSISTANCE"

    if state.status == "READY_TO_NAVIGATE":
        return "NAVIGATE_TO_DESTINATION"

    if state.status == "READY_FOR_NEXT_WAYPOINT":
        return "SEND_NEXT_WAYPOINT"

    if state.status == "REPLANNING":
        return "REQUEST_REPLAN"

    if state.status == "NAVIGATING":
        return "WAIT_FOR_FEEDBACK"

    if state.status == "YIELDED":
        return "WAIT"

    if state.status == "ARRIVED":
        return "FINISH"
    
    if state.status == "ROUTE_READY":
        return "SEND_NEXT_WAYPOINT"

    if state.status == "WAITING_FOR_CLARIFICATION":
        return "WAIT_FOR_CLARIFICATION"

    if state.status == "UNKNOWN_DESTINATION":
        return "CANNOT_NAVIGATE"

    if state.status in ("ERROR", "STOPPED"):
        return "SAFE_STOP"

    return "WAIT"

def assign_route(
    state: AgentState,
    route: RoutePlan,
) -> AgentState:
    """
    Store a newly generated route in the agent state.
    """

    state.route_id = route.route_id

    state.current_route = [
        waypoint.waypoint_id
        for waypoint in route.waypoints
    ]

    state.waypoint_index = 0
    state.total_waypoints = len(route.waypoints)

    state.replan_required = False
    state.status = "ROUTE_READY"

    return state

def request_initial_route(
    state: AgentState,
    start_location: str,
) -> AgentState:
    """
    Request the initial route for the visitor's destination.
    """

    if state.destination is None:
        state.status = "ERROR"
        return state

    route = plan_route(
        start=start_location,
        destination=state.destination,
        blocked_locations=state.blocked_locations,
    )

    if route is None:
        state.status = "NO_ROUTE_AVAILABLE"
        state.human_assistance_required = True
        return state

    return assign_route(state, route)

def get_next_waypoint(state: AgentState) -> str | None:
    """
    Return the next waypoint the robot should move toward.
    """

    if state.waypoint_index >= len(state.current_route):
        return None

    return state.current_route[state.waypoint_index]

def create_next_waypoint_command(state: AgentState):
    """
    Create the navigation command for the next waypoint
    in the active route.
    """

    waypoint_id = get_next_waypoint(state)

    if waypoint_id is None:
        state.status = "ARRIVED"
        return False, None

    if state.route_id is None:
        state.status = "ERROR"
        return False, None

    return create_navigation_command(
        state=state,
        route_id=state.route_id,
        waypoint_id=waypoint_id,
        waypoint_index=state.waypoint_index,
        total_waypoints=state.total_waypoints,
    )
    
def request_replan(
    state: AgentState,
    blocked_location: str,
) -> AgentState:
    """
    Request an alternative route after navigation
    reports that part of the current route is blocked.
    """

    # Remember this location should be avoided
    state.blocked_locations.add(blocked_location)

    if state.destination is None:
        state.status = "ERROR"
        return state

    if state.current_waypoint is None:
        state.status = "ERROR"
        return state

    route = replan_route(
        current_location=state.current_waypoint,
        destination=state.destination,
        blocked_locations=state.blocked_locations,
    )

    if route is None:
        state.status = "NO_ROUTE_AVAILABLE"
        state.human_assistance_required = True
        return state

    return assign_route(state, route)
