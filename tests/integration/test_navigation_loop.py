import pytest
from src.navigation.command_validator import validate_agent_command
from src.control.safety_supervisor import SafetySupervisor
from src.control.adapters.sim_adapter import SimAdapter
from src.navigation.route_evaluator import RouteEvaluator

def test_full_navigation_loop():
    """
    This is our grand rehearsal! We simulate one complete loop:
    Agent -> Validator -> Safety -> Adapter -> Simulator -> Evaluator -> Agent Feedback
    """
    
    # 1. Bring in all our team members
    supervisor = SafetySupervisor()
    adapter = SimAdapter()
    evaluator = RouteEvaluator()
    
    # 2. The Agent sends a raw dictionary request
    raw_agent_request = {
        "schema_version": "1.0",
        "run_id": "run_001",
        "command_id": "cmd_001",
        "destination": "radiology",
        "route_id": "route_01",
        "waypoint_id": "wp_1",
        "waypoint_index": 0,
        "total_waypoints": 5,
        "action": "MOVE_TO_WAYPOINT",
        "max_duration_s": 1.0,
        "confidence": 0.95
    }
    
    # 3. The Bouncer checks the request
    is_valid, agent_cmd = validate_agent_command(raw_agent_request)
    assert is_valid is True, "The bouncer incorrectly rejected a good command!"
    
    # 4. We peek at the simulator to see what the environment looks like right now
    obs = adapter.get_observation()
    
    # 5. The Driving Instructor checks safety against the observation and converts the command
    robot_cmd = supervisor.evaluate_command(agent_cmd, obs)
    assert robot_cmd.action == "MOVE", "The instructor should have allowed movement"
    
    # 6. We hand the safe command to our Delivery Driver (the Adapter)
    success = adapter.send_command(robot_cmd)
    assert success is True
    
    # 7. We simulate time passing, and pull a fresh observation from the simulator
    new_obs = adapter.get_observation()
    
    # 8. The Examiner looks at what happened and writes a feedback report
    # (Since our fake adapter doesn't actually move the robot yet, it won't say WAYPOINT_REACHED, 
    # it will just say EXECUTED because we are theoretically still 'driving').
    was_stopped_by_safety = (robot_cmd.action == "STOP")
    feedback = evaluator.evaluate_progress(agent_cmd, new_obs, was_stopped_by_safety)
    
    # 9. Proof that the whole loop worked seamlessly!
    assert feedback.status == "EXECUTED"
    
    # If we made it to the end without crashing, it works!