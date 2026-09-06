from agent import (
    handle_destination_decision,
    decide_next_action,
    request_initial_route,
    handle_navigation_feedback,
)

from navigation.schemas import NavigationFeedback


def test_navigate_decision_creates_goal():

    decision = {
        "intent": "NAVIGATE",
        "destination_type": "DEPARTMENT",
        "destination": "radiology",
        "ward_number": None,
        "confidence": 0.95,
    }

    state = handle_destination_decision(decision)

    assert state.destination == "radiology"
    assert state.status == "READY_TO_NAVIGATE"

    action = decide_next_action(state)

    assert action == "REQUEST_ROUTE"
    
def test_clarification_does_not_navigate():

    decision = {
        "intent": "CLARIFY",
        "destination": None,
    }

    state = handle_destination_decision(decision)

    assert state.destination is None
    assert state.status == "WAITING_FOR_CLARIFICATION"

    action = decide_next_action(state)

    assert action == "WAIT_FOR_CLARIFICATION"
    
def test_agent_can_request_route():

    decision = {
        "intent": "NAVIGATE",
        "destination_type": "DEPARTMENT",
        "destination": "radiology",
        "ward_number": None,
        "confidence": 0.95,
    }

    state = handle_destination_decision(decision)

    state = request_initial_route(
        state,
        start_location="lobby",
    )

    assert state.route_id is not None

    assert state.current_route == [
        "corridor_a",
        "corridor_b",
        "corridor_c",
        "radiology",
    ]

    assert state.waypoint_index == 0
    assert state.total_waypoints == 4
    assert state.status == "ROUTE_READY"
    
def test_agent_preserves_goal_when_route_blocked():

    decision = {
        "intent": "NAVIGATE",
        "destination_type": "DEPARTMENT",
        "destination": "radiology",
        "ward_number": None,
        "confidence": 0.95,
    }

    state = handle_destination_decision(decision)

    state = request_initial_route(
        state,
        start_location="lobby",
    )

    state.current_waypoint = "corridor_a"

    feedback = NavigationFeedback(
        run_id=state.run_id,
        command_id="test_command",
        status="BLOCKED",
        current_waypoint="corridor_a",
        target_waypoint="corridor_b",
        distance_to_target_m=3.0,
        obstacle_detected=True,
        obstacle_distance_m=1.0,
        replan_required=True,
        human_assistance_required=False,
        reason="Corridor B blocked.",
        recommended_next_action="REPLAN",
    )

    state = handle_navigation_feedback(
        state,
        feedback,
    )

    assert state.destination == "radiology"

    assert state.status == "REPLANNING"

    assert state.replan_required is True

    assert decide_next_action(state) == "REQUEST_REPLAN"