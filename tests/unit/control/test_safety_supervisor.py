import time
from src.navigation.schemas import AgentNavigationCommand, SimulatorObservation, Pose, Velocity
from src.control.safety_supervisor import SafetySupervisor

def test_safety_clamps_duration():
    supervisor = SafetySupervisor()
    
    # Arrange: AI student asks to drive forward for 100 seconds! (Very dangerous)
    agent_cmd = AgentNavigationCommand(
        run_id="run_1", command_id="cmd_1", destination="radiology", 
        route_id="r1", waypoint_id="wp_1", waypoint_index=0, total_waypoints=2,
        action="MOVE_TO_WAYPOINT", max_duration_s=100.0, confidence=0.9
    )
    
    # Arrange: A perfect, safe observation of the environment right now
    obs = SimulatorObservation(
        run_id="run_1", command_id="cmd_0", timestamp_ms=int(time.time() * 1000),
        sim_time_s=1.0, robot_status="RUNNING",
        pose=Pose(x_m=0.0, y_m=0.0, yaw_rad=0.0),
        velocity=Velocity(vx_mps=0.0, vy_mps=0.0, wz_rps=0.0),
        heading_error_rad=0.0, obstacles=[]
    )
    
    # Act: The instructor evaluates the request
    robot_cmd = supervisor.evaluate_command(agent_cmd, obs)
    
    # Assert: The instructor forced the duration down to the safe 1.0s maximum
    assert robot_cmd.action == "MOVE"
    assert robot_cmd.duration_s == 1.0 
    assert robot_cmd.safety_checked is True

def test_safety_stops_for_obstacle():
    supervisor = SafetySupervisor()
    
    # Arrange: AI wants to move forward
    agent_cmd = AgentNavigationCommand(
        run_id="run_1", command_id="cmd_1", destination="radiology", 
        route_id="r1", waypoint_id="wp_1", waypoint_index=0, total_waypoints=2,
        action="MOVE_TO_WAYPOINT", max_duration_s=1.0, confidence=0.9
    )
    
    # Arrange: Oh no! A cart blocks the path just 0.2 meters away!
    obs = SimulatorObservation(
        run_id="run_1", command_id="cmd_0", timestamp_ms=int(time.time() * 1000),
        sim_time_s=1.0, robot_status="RUNNING",
        pose=Pose(x_m=0.0, y_m=0.0, yaw_rad=0.0),
        velocity=Velocity(vx_mps=0.0, vy_mps=0.0, wz_rps=0.0),
        heading_error_rad=0.0, 
        obstacles=[{"id": "cart", "distance_m": 0.2, "blocking": True}]
    )
    
    # Act
    robot_cmd = supervisor.evaluate_command(agent_cmd, obs)
    
    # Assert: The instructor slammed the brakes! The command was changed to STOP.
    assert robot_cmd.action == "STOP"
    assert robot_cmd.vx_mps == 0.0