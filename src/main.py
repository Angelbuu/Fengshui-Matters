import time
import json
from src.navigation.command_validator import validate_agent_command
from src.control.safety_supervisor import SafetySupervisor
from src.navigation.route_evaluator import RouteEvaluator
from src.control.adapters.sim_adapter import SimAdapter

def run_navigation_loop():
    print("=== STARTING NAVIGATION PIPELINE ===")
    
    # 1. Initialize your components
    adapter = SimAdapter(mode="fake")
    safety_gate = SafetySupervisor()
    evaluator = RouteEvaluator()
    
    # 2. Mock a raw JSON request exactly as the Agent Lead should send it
    raw_agent_command = {
        "schema_version": "1.0",
        "run_id": "run-001",
        "command_id": "cmd-001",
        "destination": "radiology",
        "route_id": "route-001",
        "waypoint_id": "wp_reception_exit",
        "waypoint_index": 0,
        "total_waypoints": 3,
        "action": "MOVE_TO_WAYPOINT",
        "max_duration_s": 1.0,
        "confidence": 0.95,
        "announcement": "Proceeding"
    }
    
    print("\n[Step 1] Validating raw agent command...")
    is_valid, command_or_error = validate_agent_command(raw_agent_command)
    if not is_valid:
        print("Validation Failed:", command_or_error)
        return
    agent_cmd = command_or_error
    print("-> Validation Passed! Converted to Pydantic AgentNavigationCommand.")
    
    print("\n[Step 2] Fetching initial observation from simulator...")
    obs = adapter.get_observation()
    
    print("\n[Step 3] Applying deterministic safety gate...")
    robot_cmd = safety_gate.evaluate_command(agent_cmd, obs)
    
    print(f"\n[Step 4] Sending low-level safe command to simulator: {robot_cmd.action} (vx={robot_cmd.vx_mps}m/s)")
    adapter.send_command(robot_cmd)
    
    print("\n[Step 5] Fetching updated observation...")
    obs = adapter.get_observation()
    # Mock the simulator physically moving closer to the waypoint
    obs.distance_to_target_m = 0.4 
    
    print("\n[Step 6] Evaluating progress against target...")
    was_stopped = (robot_cmd.action == "STOP")
    feedback = evaluator.evaluate_progress(agent_cmd, obs, was_stopped_by_safety=was_stopped)
    
    print("\n[Step 7] Final Feedback to return to Agent Architecture Lead:")
    # Pretty print the final JSON feedback
    print(feedback.model_dump_json(indent=2))
    
    print("\n=== NAVIGATION PIPELINE SUCCESS ===")

if __name__ == "__main__":
    run_navigation_loop()