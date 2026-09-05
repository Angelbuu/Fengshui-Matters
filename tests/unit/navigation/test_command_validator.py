import pytest
from src.navigation.command_validator import validate_agent_command

def test_perfect_command_is_accepted():
    # Arrange: We create a perfect, rule-following dictionary
    good_data = {
        "schema_version": "1.0",
        "run_id": "run_001",
        "command_id": "cmd_001",
        "destination": "radiology",  # Correct spelling!
        "route_id": "route_01",
        "waypoint_id": "wp_1",
        "waypoint_index": 0,
        "total_waypoints": 5,
        "action": "MOVE_TO_WAYPOINT",
        "max_duration_s": 1.0,
        "confidence": 0.95
    }
    
    # Act: We run our new function
    is_valid, result = validate_agent_command(good_data)
    
    # Assert: We prove that it worked
    assert is_valid is True
    assert result.destination == "radiology" # It became a real object!

def test_bad_destination_is_rejected():
    # Arrange: We create a dictionary with a typo in the destination
    bad_data = {
        "schema_version": "1.0",
        "run_id": "run_001",
        "command_id": "cmd_002",
        "destination": "x-ray",  # WRONG! Our schema only allows "radiology"
        "route_id": "route_01",
        "waypoint_id": "wp_1",
        "waypoint_index": 0,
        "total_waypoints": 5,
        "action": "MOVE_TO_WAYPOINT",
        "max_duration_s": 1.0,
        "confidence": 0.95
    }
    
    # Act: We run our new function
    is_valid, result = validate_agent_command(bad_data)
    
    # Assert: We prove that it was safely caught and blocked
    assert is_valid is False
    assert "Command rejected" in result